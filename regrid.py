import sys, os, glob, pdb
import numpy as np
import shapefile 
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import multiprocessing
import pickle 
import pandas
from skimage import feature, measure
import scipy
import itertools

########################################
def star_outpolygon_gridin_intersection(param):
    return outpolygon_gridin_intersection(*param)


#---------------------------------------
def outpolygon_gridin_intersection(i_out_polygon, out_polygon , gridin_x,  gridin_y, gridin_polygon):
  
    out = []
    dx_in = gridin_x[1:,1:] - gridin_x[:-1,:-1]
    dy_in = gridin_y[1:,1:] - gridin_y[:-1,:-1]

    #first check for gridin reso < gridout reso
    idx1 = np.where( ( (gridin_x[:-1,:-1] - out_polygon.bounds[0]) >= -1*dx_in ) & ( (gridin_x[1:,1:] - out_polygon.bounds[2]) <= dx_in ) & 
                     ( (gridin_y[:-1,:-1] - out_polygon.bounds[1]) >= -1*dy_in ) & ( (gridin_y[1:,1:] - out_polygon.bounds[3]) <= dy_in ) )
    
    #then the opposite
    points = np.dstack(out_polygon.exterior.coords.xy)[0]
    idx2 = [[],[]]
    for point in points[:-1]:
        idx_ = np.where(
                        ((gridin_x[:-1,:-1]-point[0]) <= 1.e-6) & ((gridin_x[1:,1:] - point[0]) >= -1.e-6) & 
                        ((gridin_y[:-1,:-1]-point[1]) <= 1.e-6) & ((gridin_y[1:,1:] - point[1]) >= -1.e-6) 
                       )
        if len(idx_[0]>0):
            idx2[0].append(idx_[0][0])
            idx2[1].append(idx_[1][0])
    idx2[0] = np.array(idx2[0])
    idx2[1] = np.array(idx2[1])
    
    if idx1[0].shape[0] > idx2[0].shape[0] :
        ii_img_polygon_l = idx1[0].min()
        ii_img_polygon_u = idx1[0].max()
        jj_img_polygon_l = idx1[1].min() 
        jj_img_polygon_u = idx1[1].max() 
   
    elif (idx2[0].shape[0] != 0 ):
        ii_img_polygon_l = idx2[0].min()
        ii_img_polygon_u = idx2[0].max()
        jj_img_polygon_l = idx2[1].min()
        jj_img_polygon_u = idx2[1].max()
    else:
        return out 


    idxi = np.arange(ii_img_polygon_l,ii_img_polygon_u+1)
    idxj = np.arange(jj_img_polygon_l,jj_img_polygon_u+1)
    for iii,jjj in itertools.product(idxi, idxj):
        try:
            intersection = out_polygon.intersection(gridin_polygon[iii,jjj])
        except: 
            pdb.set_trace()
            
        if intersection.area!=0:
            out.append([iii,jjj,i_out_polygon,intersection.area])

    return out



########################################
def get_polygon_from_grid(grid_i_in, grid_j_in, flag_add_1extra_rowCol=False):
   
    if flag_add_1extra_rowCol:
        grid_i = np.zeros([grid_i_in.shape[0]+1,grid_i_in.shape[1]+1])
        grid_i[:-1,:-1] = grid_i_in
        grid_i[-1,: ] = grid_i[-2,:] + (grid_i[-2,:]-grid_i[-3,:])
        grid_i[ :,-1] = grid_i[ :,-2] + (grid_i[:,-2]-grid_i[:,-3])

        grid_j = np.zeros([grid_i_in.shape[0]+1,grid_i_in.shape[1]+1])
        grid_j[:-1,:-1] = grid_j_in
        grid_j[-1,:] = grid_j[-2,:] + (grid_j[-2,:]-grid_j[-3,:])
        grid_j[ :,-1] = grid_j[ :,-2] + (grid_j[:,-2]-grid_j[:,-3])
        
    else:
        grid_i = grid_i_in
        grid_j = grid_j_in

    ni, nj = grid_i.shape[0]-1,grid_i.shape[1]-1 
    img_polygons = []
    
    for ii,jj in itertools.product(range(ni), range(nj)):
        pts = [ [ grid_i[ii,  jj  ],grid_j[ii,  jj  ] ], \
                [ grid_i[ii+1,jj  ],grid_j[ii+1,jj  ] ], \
                [ grid_i[ii+1,jj+1],grid_j[ii+1,jj+1] ], \
                [ grid_i[ii  ,jj+1],grid_j[ii  ,jj+1] ], \
              ]
        img_polygons.append(Polygon(pts))
    img_polygons = np.array(img_polygons).reshape(ni,nj)

    return img_polygons, grid_i, grid_j



########################################
def grid2grid_pixel_match(regrid_name,                         \
                          gridin_x,  gridin_y, gridin_polygon, \
                          gridout_polygon ,                    \
                          wkdir,                               \
                          flag_parallel=False, flag_freshstart=False):

    '''
    grid data using polygon intersection between native grid and new grid.
    it uses a loop over the new grid and get each intersection with the native grid
    '''
    niin, njin = gridin_polygon.shape
    niout,njout = gridout_polygon.shape

    in_out_grid_list = np.empty((niin*njin, 0)).tolist()
    out_in_grid_list = np.empty((niout*njout, 0)).tolist()

    #match triangles and image pixels
    #-----------
    if (flag_freshstart) | (not os.path.isfile(wkdir+ regrid_name + '.p' )) : 
        #print '   match triangles and image pixels'
       
        params = []
        for i_out_polygon, out_polygon  in enumerate(gridout_polygon.flatten()):
            params.append([ i_out_polygon ,out_polygon , gridin_x,  gridin_y, gridin_polygon]) 

        if flag_parallel:
            # set up a pool to run the parallel processing
            cpus = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=cpus)

            # then the map method of pool actually does the parallelisation  
            results = pool.map(star_outpolygon_gridin_intersection, params)
            pool.close()
            pool.join()
           
        else:
            results = []
            for i_param, param in enumerate(params):
                print '{:5.2f}\r'.format(100.*i_param/(niout*njout)),
                sys.stdout.flush()
                results.append(outpolygon_gridin_intersection(*param))

        for out in results:
            for iii,jjj,i_tri,grid_img_intersection_area_m2 in out:
                
                idx_img_pixel = np.ravel_multi_index([[iii],[jjj]],(niin,njin))[0]
                in_out_grid_list[idx_img_pixel].append([i_tri,grid_img_intersection_area_m2])

                out_in_grid_list[i_tri].append([(iii,jjj),grid_img_intersection_area_m2])

        pickle.dump([in_out_grid_list,out_in_grid_list], open( wkdir+ regrid_name + '.p', "wb" ) )
    
    
    else:
        #print '   load triangle img pixel lookup table'
        in_out_grid_list,out_in_grid_list = pickle.load(open( wkdir+regrid_name+'.p', 'rb'))


    return in_out_grid_list,out_in_grid_list



########################################
def map_data(out_in_grid_list,out_dimensions,data_in,flag='average',gridReso_in=None):

    nx2,ny2 = out_dimensions
    data_out            = np.zeros(nx2*ny2)
    data_out_pixel_area = np.zeros(nx2*ny2)

    for i_outpolygon in range(nx2*ny2):
        for idx_in, area_intersect in out_in_grid_list[i_outpolygon]:
            if flag == 'average':
                data_out_pixel_area[i_outpolygon] += area_intersect 
                data_out[i_outpolygon]            += area_intersect * data_in[idx_in]
           

            elif flag == 'sum':
                data_out[i_outpolygon]            += area_intersect/gridReso_in[idx_in] * data_in[idx_in]
                data_out_pixel_area[i_outpolygon] += area_intersect 
                
            elif flag == 'max':
                if len(data_out[i_outpolygon])>1:
                    data_out[i_outpolygon]             = max([data_out[i_outpolygon],data_in[idx_in].max()])
                else:
                    data_out[i_outpolygon]             = data_in[idx_in].max()
                data_out_pixel_area[i_outpolygon] += area_intersect 

            else: 
                print 'check flag in regrid'
                sys.exit()
    
    if flag == 'average':
        idx = np.where(data_out_pixel_area!=0)
        data_out[idx] /= data_out_pixel_area[idx]
        idx = np.where(data_out_pixel_area==0)
        data_out[idx] = -999
    
    return data_out.reshape(nx2,ny2),data_out_pixel_area.reshape(nx2,ny2)


###################################
if __name__ == '__main__':
###################################

    #set example input data
    #######################

    #gridin
    nx, ny = 100, 100
    Lx, Ly = 101, 101
    dxin,dyin = 1.*Lx/nx,1.*Ly/ny
    gridin_y, gridin_x = np.meshgrid(np.linspace(0,Ly,ny+1),np.linspace(0,Lx,nx+1)) 
    data_in = np.zeros([nx,ny])
    data_in[25:-25,25:-25] = 1
    gridin_res = dxin*dyin*np.ones_like(data_in)

    #gridout
    nx2, ny2 = 150, 150
    Lx2, Ly2 = 90, 90
    dxout,dyout = 1.*Lx2/nx2,1.*Ly2/ny2
    gridout_y, gridout_x = np.meshgrid(np.linspace(0,Ly2,ny2+1),np.linspace(0,Lx2,nx2+1)) 



    #match pixels from in and out grid
    #######################
    regrid_name = 'test'
    wkdir = './'
    gridin_polygon, gridin_xx, gridin_yy    = get_polygon_from_grid(gridin_x,  gridin_y )
    gridout_polygon, gridout_xx, gridout_yy = get_polygon_from_grid(gridout_x, gridout_y)

    in_out_grid_list, out_in_grid_list = grid2grid_pixel_match(regrid_name,                         \
                                                               gridin_xx,  gridin_yy, gridin_polygon, \
                                                               gridout_polygon,                     \
                                                               wkdir, flag_parallel=False, flag_freshstart=True)

    
    #map data
    #######################
    data_out, data_out_pixel_area = map_data(out_in_grid_list,[gridout_xx.shape[0]-1,gridout_xx.shape[1]-1],data_in,flag='sum',gridReso_in=gridin_res)


    print data_in.sum(), data_out.sum()

    #plot
    #######################
    plt.figure(1,figsize=(15,6))
    ax = plt.subplot(131)
    ax.imshow(data_in.T,origin='lower',extent=(gridin_x.min(),gridin_x.max(),gridin_y.min(),gridin_y.max()) )
    ax = plt.subplot(132)
    ax.imshow(np.ma.masked_where(data_out==-999,data_out).T,origin='lower',extent=(gridout_x.min(),gridout_x.max(),gridout_y.min(),gridout_y.max()) )
    ax = plt.subplot(133)
    ax.imshow(np.ma.masked_where(data_out==-999,data_out_pixel_area).T,origin='lower',extent=(gridout_x.min(),gridout_x.max(),gridout_y.min(),gridout_y.max()) )
    plt.show()

