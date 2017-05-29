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
    return triangle_img_pixel_intersection(*param)


#---------------------------------------
def outpolygon_gridin_intersection(i_out_polygon, out_polygon , gridin_x,  gridin_y, gridin_polygon):
  
    out = []
    
    #first check for gridin reso < gridout reso
    idx1 = np.where( (gridin_x >= out_polygon.bounds[0]) & (gridin_x <= out_polygon.bounds[2]) & 
                    (gridin_y >= out_polygon.bounds[1]) & (gridin_y <= out_polygon.bounds[3]) )
    
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
        pdb.set_trace()
        return out 


    idxi = np.arange(ii_img_polygon_l,ii_img_polygon_u+1)
    idxj = np.arange(jj_img_polygon_l,jj_img_polygon_u+1)
    for iii,jjj in itertools.product(idxi, idxj):
        intersection = out_polygon.intersection(gridin_polygon[iii,jjj])
        if intersection.area!=0:
            out.append([iii,jjj,i_out_polygon,intersection.area])

    return out



########################################
def get_polygon_from_grid(grid_i, grid_j):
   
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

    return img_polygons



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



###################################
if __name__ == '__main__':
###################################

    #set example input data
    #######################

    #gridin
    nx, ny = 100, 100
    Lx, Ly = 100, 100
    gridin_y, gridin_x = np.meshgrid(np.linspace(0,Ly+Ly/ny,ny+1),np.linspace(0,Lx+Lx/nx,nx+1)) 
    data_in = np.zeros_like(gridin_x)
    data_in[25:-25,25:-25] = 1

    #gridout
    nx2, ny2 = 100, 100
    Lx2, Ly2 = 50, 50
    gridout_y, gridout_x = np.meshgrid(np.linspace(0,Ly2+Ly2/ny2,ny2+1),np.linspace(0,Lx2+Lx2/nx2,nx2+1)) 



    #match pixels from in and out grid
    #######################
    regrid_name = 'test'
    wkdir = './'
    gridin_polygon  = get_polygon_from_grid(gridin_x,  gridin_y )
    gridout_polygon = get_polygon_from_grid(gridout_x, gridout_y)

    in_out_grid_list, out_in_grid_list = grid2grid_pixel_match(regrid_name,                         \
                                                               gridin_x,  gridin_y, gridin_polygon, \
                                                               gridout_polygon,                     \
                                                               wkdir, flag_parallel=False, flag_freshstart=True)

    
    #map data
    #######################
    data_out            = np.zeros(nx2*ny2)
    data_out_pixel_area = np.zeros(nx2*ny2)

    for i_outpolygon in range(nx2*ny2):
        for idx_in, area_intersect in out_in_grid_list[i_outpolygon]:
            data_out_pixel_area[i_outpolygon] += area_intersect 
            data_out[i_outpolygon]            += area_intersect * data_in[ idx_in ]
    idx = np.where(data_out_pixel_area!=0)
    data_out[idx] /= data_out_pixel_area[idx]
    idx = np.where(data_out_pixel_area==0)
    data_out[idx] = -999
    data_out = data_out.reshape(nx2,ny2)



    #plot
    #######################
    plt.figure(1,figsize=(15,6))
    ax = plt.subplot(131)
    ax.imshow(data_in.T,origin='lower',extent=(gridin_x.min(),gridin_x.max(),gridin_y.min(),gridin_y.max()) )
    ax = plt.subplot(132)
    ax.imshow(np.ma.masked_where(data_out==-999,data_out).T,origin='lower',extent=(gridout_x.min(),gridout_x.max(),gridout_y.min(),gridout_y.max()) )
    ax = plt.subplot(133)
    ax.imshow(np.ma.masked_where(data_out==-999,data_out_pixel_area.reshape(nx2,ny2)).T,origin='lower',extent=(gridout_x.min(),gridout_x.max(),gridout_y.min(),gridout_y.max()) )
    plt.show()

