import numpy as np
import open3d as o3d
import time
import math
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.transform import Rotation as R

def media_z(plano):
    soma_face = 0
    #media do z do plano -> plano é um np array com os pontos da point cloud 
    for x in plano:
        #print("Z: ",x[2],"\n")
        soma_face += x[2]
    
    z_av = soma_face / (len(plano))    #plano.size dá o tamanaho x3 porque cada coordendada conta como 1

    return z_av

def RANSAC_pcd(pcd,distance_threshold,ransac_n,num_iterations): 
    # distance_thresold = 15, ransac_n = 5, num_iterations = 1000
    #plane model retorna a equação do plano que o ransac detetou, inliers retorna os indices num array da pcd usada para o ransac dos pontos dos inliers
    plane_model , inliers = pcd.segment_plane(distance_threshold = distance_threshold, ransac_n = ransac_n, num_iterations = num_iterations)
    #Plano
    inlier_cloud = pcd.select_by_index(inliers)                 # inlier_cloud- coordenadas da pcd dos inliers
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)   # outlier_cloud - coordenadas da pcd dos outliers
    outlier_cloud.paint_uniform_color([0, 1.0, 0])             # pinta os pontos da outlier_cloud todos de verde neste caso, formato RGB
    inlier_cloud.paint_uniform_color([1.0, 0, 0])              # pinta os pontos da inlier_cloud todos de vermelho neste caso, formato RGB
    return plane_model, inliers, inlier_cloud, outlier_cloud

def downsamp_pcd(pcd, voxel_size):
    downpcd=pcd.voxel_down_sample(voxel_size=voxel_size)
    return downpcd

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    #     rotations = np.vstack([
    #         np.cos(angles),
    #         -np.sin(angles),
    #         np.sin(angles),
    #         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

def create_pcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def visual_pcd(self):
    o3d.visualization.draw_geometries_with_editing([self])
  
def visual2_pcd(self, self1):
    o3d.visualization.draw_geometries([self,self1])    

def get_vector_angle(u,v):
    #print(u,v)
    uv=np.dot(u,v)
    ang = np.arccos(uv)
    ang = ang*180/np.pi
    if (ang > 90):
        ang = 180 - ang
    return (ang)

def get_plane_distance(n,p):
    d=np.dot(n,p)
    return (d)

def get_distance_group(dist,bin_indexes,dist_th):
    planebins={} #bins
    distValues={} #refPoints
    planebins[0]=[bin_indexes[0]]
    distValues[0]=dist[0]
    u=dist[0]
    
    for i in np.arange(1,len(bin_indexes)):
        v=dist[i]
        for y in distValues: #Add the new member in the matching group
            u=distValues[y]
            if abs(u-v) <= dist_th:
                planebins[y] += [bin_indexes[i]]
                break
        else: #Create a new group center
            new_group = len(distValues)
            distValues[new_group] = v
            planebins[new_group]=[bin_indexes[i]]
        #print(planebins)
    return (planebins, distValues)

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def color_pcd(downpcd,s_bins,colors):
    for i in s_bins:
        #np.asarray(downpcd.colors)[(s_bins[i])[1:], :] = colors[i]
        np.asarray(downpcd.colors)[(s_bins[i])[1:], :] = [np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)]

def bin_info(bins,refPoint):
    for k in bins:
        print("Bins: "+str(k)+"   RefPoint: "+str(refPoint[k])+"   Dimension: "+str(len(bins[k])))

def bin_sort(bins,refPoint):
    order=[]
    for k in sorted(bins, key=lambda k: len(bins[k]), reverse=True):
        order.append(k)
    s_bins={}
    s_refPoint={}
    h=0
    for d in order:
        #print("Cluster: "+str(d)+"   RefPoint: "+str(refPoint[d])+"   Dimension: "+str(len(bins[d])))
        s_bins[h]=bins[d]
        s_refPoint[h]=refPoint[d]
        h+=1
    return s_bins,s_refPoint

def bin_filter(s_bins,s_refPoint,percent):
    for g in range(len(s_bins.keys())):
        if len(s_bins[g]) <= percent*len(s_bins[0]):
            del s_bins[g]
            del s_refPoint[g]
    return s_bins,s_refPoint

def plane_obb(pcd,Z):  

    def find_bound(xyz):
        i=0
        x_max=0
        x_min=2000
        y_max=0
        y_min=2000
        z_max=0
        z_min=2000
        for p in xyz:
            #FIND XMAX e XMIN
            if p[0] > x_max:
               x_max=p[0]
            if p[0] < x_min:
               x_min=p[0]
            #FIND YMAX e YMIN
            if p[1] > y_max:
                y_max=p[1]
            if (p[1] < y_min):
                y_min=p[1]
            #FIND ZMAX e ZMIN
            if p[2] > z_max:
                z_max=p[2]
            if (p[2] < z_min):
                z_min=p[2]
        x=[x_min,x_max]
        y=[y_min,y_max]
        z=[z_min,z_max]
        #print("X   Max=",x_max, "\tMin=",x_min)
        #print("Y   Max=",y_max, "\tMin=",y_min)
        #print("Z   Max=",z_max, "\tMin=",z_min)
        return x,y,z

    def min_max(e):
        e_min=e[0]
        e_max=e[1]
        return e_min, e_max

    def corners(x,y,z,xyz):
        x_min,x_max=min_max(x)
        y_min,y_max=min_max(y)
        z_min,z_max=min_max(z)
        i=0
        for p in xyz:
            if p[0] == x_max:
                C4=p
                #C4=np.asarray((C4[0],C4[1],z[2]))
            if p[0] == x_min:
                C2=p
                #C2=np.asarray((C2[0],C2[1],z[2]))
            if p[1] == y_max:
                C3=p
                #C3=np.asarray((C3[0],C3[1],z[2]))
            if (p[1] == y_min):
                C1=p
                #C1=np.asarray((C1[0],C1[1],z[2]))
        return C1,C2,C3,C4

    def obb_pcd(pcd):
        obb=o3d.geometry.OrientedBoundingBox.get_oriented_bounding_box(pcd)           
        return obb

    def cornersplus(C1,C2,C3,C4,Z):
        C5=np.asarray((C1[0],C1[1],z)) #z[1]= z_max e.g. z=[z_min,z_max]
        C6=np.asarray((C2[0],C2[1],z))
        C7=np.asarray((C3[0],C3[1],z))
        C8=np.asarray((C4[0],C4[1],z))           
        list= [C1,C2,C3,C4,C5,C6,C7,C8]
        point_list=np.asarray(list)
        return point_list

    xyz=np.asarray(pcd.points)
    x,y,z=find_bound(xyz) # Encontra máximos
    C1,C2,C3,C4=corners(x,y,z,xyz) # Encontra os primeiros 4 vertices (os de cima)
    corners=cornersplus(C1,C2,C3,C4,z)# Faz os restantes 4 vertices (os de baixo)
    corner=create_pcd(corners) #cria pcd com esses pontos           
    obb = obb_pcd(corner) #Faz o obb desses pontos
    return obb

def DBSACN_pcd(pcd,eps,min_points):
    
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    #np.set_printoptions(threshold=sys.maxsize)
    #print (labels.size)
    #print (labels)
    return labels

def get_Z_min(pcd):
    # funçao que retorna o zmin de uma pdc sendo o input a pcd
    zmin=999
    index=0
    tamanho=len(np.asarray(downpcd.points))  

    while index< tamanho:
        valor_um=(np.asarray(downpcd.points))[index]

        if ( valor_um[2]<zmin ): 
            zmin=valor_um[2]

        index=index+1    

    return zmin     

inicio=time.time()
#   Read pointcloud


pcd = o3d.io.read_point_cloud("testpcd.pcd")
#pcd = o3d.io.read_point_cloud("Imagens_Proj\Varias Caixas\Frame373.ply")
#pcd = o3d.io.read_point_cloud("duas_caixas.pcd")
downpcd = pcd.voxel_down_sample(voxel_size=5)  #25

#o3d.visualization.draw_geometries([downpcd])

#downpcd.paint_uniform_color([0.8, 0.8, 0.8])
cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
downpcd = downpcd.select_by_index(ind)
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=300)) # 20 e 200 standard
downpcd.normalize_normals()
normals= np.asarray(downpcd.normals)

##########
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300, origin=[0,0,0])
##mesh_r = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=[50,50,50])
#R = mesh.get_rotation_matrix_from_xyz((0, 0, 0))
#rotMatrix=pcd.get_rotation_matrix_from_xyz((0, 0, 0))
#print(R)
#print(rotMatrix)
#mesh_r = mesh
#downpcd.rotate(mesh.get_rotation_matrix_from_xyz((0,250*np.pi/360,0)))
o3d.visualization.draw_geometries([downpcd, mesh])
################


ang_th=25
dist_th=35
npoints_th=500

# Angular clusters
#lista com os valores do vetor normal de todos os pontos

normals= np.asarray(downpcd.normals)
bins={}
refPoint={}
bins[0]=[0]
refPoint[0]=list(normals[0])

u=normals[0]
for i in np.arange(1,len(normals)):
    v=normals[i]
    for key in refPoint: #Add the new member in the matching group
        u=refPoint[key]
        if get_vector_angle(u,v) <=ang_th:
            bins[key].append(i)
            break
    else: #Create a new group center
        new_bin=len(refPoint)
        refPoint[new_bin]=v
        bins[new_bin]=[i]
#print("-----------Original------------")
#bin_info(bins,refPoint)
print("-----------Sorted------------")
s_bins,s_refPoint=bin_sort(bins,refPoint)
bin_info(s_bins,s_refPoint)
print("-----------Filtered------------")
f_bins,f_refPoint=bin_filter(s_bins,s_refPoint,0.0000001) # para as boxes é 0.02
bin_info(f_bins,f_refPoint)


colors=[[0, 1, 0],[0, 0, 1],
        [0.188, 0.635, 0.243],[0.933, 0.431, 0.156],
        [0.8, 0.670, 0.086],[0.921, 0.035, 0.996],
        [0.403, 1, 0.490],[0.258, 0.878, 0.874],
        [0.490, 0.047, 0.101],[0.172, 0.180, 0.356]]
   
#visual_pcd(downpcd)

xs=[]
ys=[]
zs=[]
#----------------------------Checks for horizontal planes
for c in s_bins:
    for p in s_bins[c]:
        xs.append(normals[p][0])
        ys.append(normals[p][1])
        zs.append(normals[p][2])
    print("Média: ",np.mean(xs),"\tMaior valor: ",np.max(xs),"\tMenor valor: ",np.min(xs))
    print("Média: ",np.mean(ys),"\tMaior valor: ",np.max(ys),"\tMenor valor: ",np.min(ys))
    print("Média: ",np.mean(zs),"\tMaior valor: ",np.max(zs),"\tMenor valor: ",np.min(zs))   
    print("Break---------------------------------")
    if np.mean(zs)> 0.5:
        h_plane=c
        break
    xs.clear()
    ys.clear()
    zs.clear() 
print("Horizontals found... They are in Bin:",h_plane)
v_plane=[]
#--------------------------------------Check for vertical planes # Não está a correr
for r in s_refPoint:
    i_product=np.inner(s_refPoint[h_plane],s_refPoint[r])
    if abs(i_product) < 0.4:
        v_plane.append(r)
        #break
print("Verticals found... They are in Bin:",v_plane)
np.asarray(downpcd.colors)[(s_bins[h_plane])[1:], :] = [0,1,0]
#for v in v_plane:  
#    np.asarray(downpcd.colors)[(s_bins[v])[1:], :] = [0,0,1]

#visual_pcd(downpcd)
#----------------------------------------Tendo os indices dos planos horizontais
h_points=[]
for idx in s_bins[h_plane]: # para cada indice do ponto dentro do cluster
    p=downpcd.points[idx]
    h_points.append(p)

horizon=np.asarray(h_points) #Planos hozintais em formato nsarray

#print(np.asarray(h_points))
#PointClouTotal=o3d.geometry.PointCloud()
#PointClouTotal.points = o3d.utility.Vector3dVector(points)
#pcdThreshold = o3d.geometry.PointCloud()
#pcdThreshold.points = o3d.utility.Vector3dVector(points)

print("Size of horizontal planes points: "+str(len(horizon)))

#--------------------------------------Encontrar zmin
zmax=0
zmin=2000
for i in range(len(horizon)):
    valor_um=horizon[i]
    if ( valor_um[2]<zmin ): 
        zmin=valor_um[2]
    if ( valor_um[2]>zmax ):
        zmax=valor_um[2]
print(zmin)

z_threshold=30
mask = horizon[:,2] > z_threshold
box_tops=o3d.geometry.PointCloud()
box_tops.points = o3d.utility.Vector3dVector(horizon[mask]) # normals and colors are unchanged
#visual_pcd(box_tops)

labels=DBSACN_pcd(box_tops,30,50)
print(labels)
# alternative
#pcd = pcd.select_by_index(np.where(points[:,2] > z_threshold)[0])



#----------Adicionar todos os indices dos pontos com a mesma label a um dicionário

tops={} #dicionario com os vários pontos correspondentes a cada caixa, com base nas labels do clustering
tops[0]=[0]
for i in np.arange(1,len(labels)):
    for k in tops:
        if labels[i] == k:
            tops[k].append(i)
            break
    else:
        if len(tops) <= max(labels):
            new_top=len(tops)
            tops[new_top]=[i]

#------------------------------"tops" contém todos os índices dos pontos pertencentes a cada face superior do objeto
#boxes_tops = box_tops.select_by_index(tops[1])
#visual_pcd(boxes_tops)

#-------------------Achar alturas respetivas dos planos
tops_h={}
listaTop=[]
lista_tops_h=[]

# inicia as listas dos planos com o chao e a das alturas
eq_face, inliers,inliers_pcd,outliers_pcd = RANSAC_pcd(downpcd,5,3,1000)
zChao=media_z(inliers_pcd.points)
lista_tops_h.append(zChao)
listaTop.append(inliers_pcd) 

centros_geometricos=[]

for t in tops:
    dummy_top = box_tops.select_by_index(tops[t])
    z_av=media_z(dummy_top.points)
    tops_h[t]=z_av
    listaTop.append(dummy_top)
    #visual_pcd(dummy_top)
    centros_geometricos.append(o3d.geometry.AxisAlignedBoundingBox.get_center(dummy_top))
    lista_tops_h.append(tops_h[t])

print(centros_geometricos)

listaCG = []
    
for t in range(len(centros_geometricos)):
    x_cg = centros_geometricos[t][0]
    y_cg = centros_geometricos[t][1]
    par_cg = np.asarray((x_cg,y_cg))
    listaCG.append(par_cg)

CG_2D = np.zeros((len(listaCG),2))

for t in range(len(listaCG)):
    CG_2D[t] = listaCG[t]

print(CG_2D)

flag=False
for t in range(len(CG_2D)-1):
    dist = math.sqrt((CG_2D[t][0] - CG_2D[t+1][0])**2 + (CG_2D[t][1] - CG_2D[t+1][1])**2)
    print(dist)
    if dist<100:
        flag=True

print(flag)

lista_line_sets=[]
lista_volumes=[]

if flag==True:
    # junta numa lista os planos das faces e as suas alturas
    listaFACES = []
    for t in range(len(listaTop)):
        face = listaTop[t]
        height = lista_tops_h[t]
        par = np.asarray((face,height))
        listaFACES.append(par)

    # ordena a lista por ordem as alturas
    listFACES_sorted = sorted(listaFACES, key=lambda x: x[1])

    # junta os troços de plano com alturas proximas (5%)
    index1=0
    index2=1
    while(index1<len(listFACES_sorted)):

        while (index2<len(listFACES_sorted)):
            if (listFACES_sorted[index1][1]>=0.95*listFACES_sorted[index2][1] and listFACES_sorted[index1][1]<=1.05*listFACES_sorted[index2][1]):
                listFACES_sorted[index1][0]=listFACES_sorted[index1][0]+listFACES_sorted[index2][0]
                del listFACES_sorted[index2]
            else:
                index2=index2+1

        index1=index1+1
        index2=index1+1

    # processamento principal
    index=len(listFACES_sorted)-1
    while(index>0):

        xyz_face = listFACES_sorted[index][0]

        # projeta os planos de cima pra baixo
        if (index<(len(listFACES_sorted)-1)):
            for t in range(len(listFACES_sorted)-index-1):
                xyz_face = xyz_face+listFACES_sorted[index+t+1][0]

        # remove ruído
        xyz_face, ind = xyz_face.remove_statistical_outlier(nb_neighbors=20,std_ratio=5.0)

        #visual_pcd(xyz_face)

        # forçar 2D
        listaXY = []
    
        for t in range(len(xyz_face.points)-1):
            x_face = xyz_face.points[t][0]
            y_face = xyz_face.points[t][1]
            par = np.asarray((x_face,y_face))
            listaXY.append(par)

        face_2D = np.zeros((len(listaXY),2))

        for t in range(len(listaXY)):
            face_2D[t] = listaXY[t]

        #print(face_2D)

        # Delaunay triangulation
        points = np.array(face_2D)
        #tri = Delaunay(points)

        # view triangulation
        #plt.triplot(points[:,0], points[:,1], tri.simplices)
        #plt.plot(points[:,0], points[:,1], 'o')
        #plt.show()

        # view edges
        hull = ConvexHull(points)
        _ = convex_hull_plot_2d(hull)
        #plt.show()

        vertices=minimum_bounding_rectangle(points)
        #print(vertices)
        plt.triplot(vertices[:,0], vertices[:,1])
        plt.plot(vertices[:,0], vertices[:,1], 'o')
        #plt.show()


        C1=np.asarray((vertices[0][0],vertices[0][1],listFACES_sorted[index][1]))
        C2=np.asarray((vertices[1][0],vertices[1][1],listFACES_sorted[index][1]))
        C3=np.asarray((vertices[2][0],vertices[2][1],listFACES_sorted[index][1]))
        C4=np.asarray((vertices[3][0],vertices[3][1],listFACES_sorted[index][1]))


        C5=np.asarray((vertices[0][0],vertices[0][1],listFACES_sorted[index-1][1])) 
        C6=np.asarray((vertices[1][0],vertices[1][1],listFACES_sorted[index-1][1]))
        C7=np.asarray((vertices[2][0],vertices[2][1],listFACES_sorted[index-1][1]))
        C8=np.asarray((vertices[3][0],vertices[3][1],listFACES_sorted[index-1][1]))


        list= [C1,C2,C3,C4,C5,C6,C7,C8]
        point_list=np.asarray(list)

        corn_list= list
        point_list=np.asarray(corn_list)
        #LINESET:
        pcdLineSet = o3d.geometry.PointCloud()
        pcdLineSet.points = o3d.utility.Vector3dVector(point_list)
        pcdLineSet.paint_uniform_color([0.0,0.0,1.0])
        #line set (usa-se em vez do obb porque o obb aparece em excesso)
        #lines = [[0,1],[0,3],[4,5],[4,7],[2,1],[2,3],[6,7],[6,5],[1,5],[0,4],[3,7],[2,6]]
        lines = [[0,1],[0,3],[4,5],[4,7],[2,1],[2,3],[6,7],[6,5],[1,5],[0,4],[3,7],[2,6],[0,2],[4,6],[1,3],[5,7]]
        colors = [[0,0,0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(point_list)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        lista_line_sets.append(line_set)
        lista_volumes.append(hull.volume*(lista_tops_h[index]-listFACES_sorted[len(listFACES_sorted)-1][1]))
        #o3d.visualization.draw_geometries([pcd,line_set])


        index=index-1


    if(len(lista_line_sets)==5):
        o3d.visualization.draw_geometries([pcd,lista_line_sets[0],lista_line_sets[1],lista_line_sets[2],lista_line_sets[3],lista_line_sets[4]])

    if(len(lista_line_sets)==4):
        o3d.visualization.draw_geometries([pcd,lista_line_sets[0],lista_line_sets[1],lista_line_sets[2],lista_line_sets[3]]) 

    if(len(lista_line_sets)==3):
        o3d.visualization.draw_geometries([pcd,lista_line_sets[0],lista_line_sets[1],lista_line_sets[2]]) 
    # retirar o lista_line_set[2] para as imagens com apenas 2 caixas

    if(len(lista_line_sets)==2):
        o3d.visualization.draw_geometries([pcd,lista_line_sets[0],lista_line_sets[1]]) 

    if(len(lista_line_sets)==1):
        o3d.visualization.draw_geometries([pcd,lista_line_sets[0]]) 
    #index=0
    #while(index<len(lista_volumes)):
    #    print("Volumes da caixa",index+1,"(dm3):")
    #    print(lista_volumes[index])
    #    index=index+1    
else:
    zmin=get_Z_min(downpcd)         

    #lista_tops_h, listaTop=juntar_planos(lista_tops_h, listaTop)
    lista_line_sets=[]
    lista_volumes=[]

    index=0

    eq_face, inliers,inliers_pcd,outliers_pcd = RANSAC_pcd(downpcd,5,3,1000)
    zChao=media_z(inliers_pcd.points)

    while(index<len(listaTop)):

        downpcd = listaTop[index]
        #o3d.visualization.draw_geometries(downpcd)

        #eq_chao, ind_inl,inl_pcd,outl_pcd = RANSAC_pcd(downpcd,5,3,1000)
        #xyz_chao = np.asarray(inl_pcd.points)
        #visual2_pcd(inl_pcd,outl_pcd)

        # topo da caixa
        eq_face, inliers,inliers_pcd,outliers_pcd = RANSAC_pcd(downpcd,5,3,1000)
        xyz_face = np.asarray(inliers_pcd.points)
        #visual2_pcd(inliers_pcd,outliers_pcd)

        # media dos z e da face
        z_av_c = zChao
        z_av_f = lista_tops_h[index]

        # forçar 2D
        listaXY = []

        for t in range(len(xyz_face)-1):
            x_face = xyz_face[t][0]
            y_face = xyz_face[t][1]
            par = np.asarray((x_face,y_face))
            listaXY.append(par)

        face_2D = np.zeros((len(listaXY),2))

        for t in range(len(listaXY)):
            face_2D[t] = listaXY[t]

        #print(face_2D)

        # Delaunay triangulation
        points = np.array(face_2D)
        #tri = Delaunay(points)

        # view triangulation
        #plt.triplot(points[:,0], points[:,1], tri.simplices)
        #plt.plot(points[:,0], points[:,1], 'o')
        #plt.show()

        # view edges
        hull = ConvexHull(points)
        _ = convex_hull_plot_2d(hull)
        #plt.show()

        vertices=minimum_bounding_rectangle(points)
        print(vertices)
        plt.triplot(vertices[:,0], vertices[:,1])
        plt.plot(vertices[:,0], vertices[:,1], 'o')
        #plt.show()


        C1=np.asarray((vertices[0][0],vertices[0][1],z_av_c)) #z[1]= z_max e.g. z=[z_min,z_max]  
        C2=np.asarray((vertices[1][0],vertices[1][1],z_av_c)) #neste caso o zminimo tem que ser o z do proprio plano e não o a seguir
        C3=np.asarray((vertices[2][0],vertices[2][1],z_av_c))
        C4=np.asarray((vertices[3][0],vertices[3][1],z_av_c))


        C5=np.asarray((vertices[0][0],vertices[0][1],z_av_f)) #z[1]= z_max e.g. z=[z_min,z_max]  
        C6=np.asarray((vertices[1][0],vertices[1][1],z_av_f)) #neste caso o zminimo tem que ser o z do proprio plano e não o a seguir
        C7=np.asarray((vertices[2][0],vertices[2][1],z_av_f))
        C8=np.asarray((vertices[3][0],vertices[3][1],z_av_f))


        list= [C1,C2,C3,C4,C5,C6,C7,C8]
        point_list=np.asarray(list)

        corn_list= list
        point_list=np.asarray(corn_list)
        #LINESET:
        pcdLineSet = o3d.geometry.PointCloud()
        pcdLineSet.points = o3d.utility.Vector3dVector(point_list)
        pcdLineSet.paint_uniform_color([0.0,0.0,1.0])
        #line set (usa-se em vez do obb porque o obb aparece em excesso)
        #lines = [[0,1],[0,3],[4,5],[4,7],[2,1],[2,3],[6,7],[6,5],[1,5],[0,4],[3,7],[2,6]]
        lines = [[0,1],[0,3],[4,5],[4,7],[2,1],[2,3],[6,7],[6,5],[1,5],[0,4],[3,7],[2,6],[0,2],[4,6],[1,3],[5,7]]
        colors = [[0,0,0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(point_list)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        lista_line_sets.append(line_set)
        lista_volumes.append(hull.volume*(z_av_f-zChao))
        #o3d.visualization.draw_geometries([pcd,line_set])

        index=index+1

    del lista_line_sets[0]
    del lista_volumes[0]

    if(len(lista_line_sets)==6):
        o3d.visualization.draw_geometries([pcd,lista_line_sets[0],lista_line_sets[1],lista_line_sets[2],lista_line_sets[3],lista_line_sets[4],lista_line_sets[5]])

    if(len(lista_line_sets)==5):
        o3d.visualization.draw_geometries([pcd,lista_line_sets[0],lista_line_sets[1],lista_line_sets[2],lista_line_sets[3],lista_line_sets[4]])

    if(len(lista_line_sets)==4):
        o3d.visualization.draw_geometries([pcd,lista_line_sets[0],lista_line_sets[1],lista_line_sets[2],lista_line_sets[3]]) 

    if(len(lista_line_sets)==3):
        o3d.visualization.draw_geometries([pcd,lista_line_sets[0],lista_line_sets[1],lista_line_sets[2]]) 
    # retirar o lista_line_set[2] para as imagens com apenas 2 caixas

    if(len(lista_line_sets)==2):
        o3d.visualization.draw_geometries([pcd,lista_line_sets[0],lista_line_sets[1]]) 

    if(len(lista_line_sets)==1):
        o3d.visualization.draw_geometries([pcd,lista_line_sets[0]]) 

    index=0
    while(index<len(lista_volumes)):
        print("Volumes da caixa",index+1,"(dm3):")
        print(lista_volumes[index])
        index=index+1    