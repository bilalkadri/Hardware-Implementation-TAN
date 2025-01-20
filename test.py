import numpy as np

dict = {"a": [[1,2]], "b": [[2,3]], "c": [[3,1]], "d": [[4,2]]}


print(type(dict))
for key, value in dict.items():
    print(key, ' : ', value)

matrix_a=dict["a"]
print("The type of matrix a is:",type(matrix_a))

print("------------Info about a-----------------")
np_array_matrix_a=np.array(matrix_a)
print("The type of  np matrix a is:",type(np_array_matrix_a))
print("The shape of np matrix b is :",np.shape(np_array_matrix_a))
print(np_array_matrix_a)

print("------------Info about b-----------------")
b=np.arange(5)
print("The type of np array b is :",type(b))
print("The shape of np array b is :",np.shape(b))
c=b+1
print(b)
print(c)
d=b*c
e=b/c
print("I am printing d:",d)
print("I am printing e:",e)
print(b)

""" delta_x = np.zeros((15,1))
print("The shape of delta_x:",np.shape(delta_x))
print(delta_x)
delta_x[[0,1,2],0] = [0.05, 0, 0]
print(delta_x) """

""" arr = np.array([[1, 2, 3], [4, 5, 6]])
arr[0,[0,1]]=[10,11]
print(arr) """

a = np.array([[1],[2],[3]])
b = np.array([[4],[5],[6]])
test=np.hstack((a,b))
""" print("I am printing test:\n",test)
print("The shape of test is:\n",np.shape(test)) """

#Ck = [np.eye(3) np.zeros(3) np.zeros(3) np.zeros(3) np.zeros(3)]
#Ck=[np.eye(3), np.zeros([3,3])] 
a=np.eye(3)
b=np.zeros([3,12])
print("I am prinitng b:",b)
c=np.ones([2,2])
Ck=np.hstack((a,b))

""" print("I am printing Ck:\n",Ck)
print("The shape of Ck is:",np.shape(Ck)) """

Rk=np.diag([1,2,3])
print("I am priting Rk:\n:",Rk)

for x in range(600, 810, 10):
  print(x)

#print(np.arccos(30*(np.pi/180)))
print('Calculating secant :')
cos_value = np.cos(30*(np.pi/180))
sec_value = np.arccos(cos_value)
print(1/cos_value)

#Multiplying a matrix by a scalar 
array1=np.array([[1,200],[3,4]])
array1_n=np.multiply(array1,0.1)
print('Maximum value of the array1 is:',np.amax(array1))
""" #phi=[[1 2 3 4 5 6]]
print(array1_n)
array2=np.array([[1],[2],[3],[4],[5],[6],[7]])
print("array2 is:",array2)
shape_array2=np.shape(array2)
print("Shape of array2 is:",shape_array2)  #(2000,1)
print("First element of shape_array2[0] is:",shape_array2[0])

print("Slicing the array2 :",array2[0:3:2,0]) """


#Appending more rows in a 2D numpy array
array2=np.array([[1,2],[4,3],[4,6]])
print("Original array2 \n",array2)
empty_array=np.empty((3,2),float)
empty_array[:,:]=100
array2_new=np.append(array2,empty_array,axis=0)
#array2=np.append(array2,np.array([[0.1,0.2,0.3]]),axis=0)
print("AFTER APPENDING A NEW ROW\n",array2_new)

#k = 0:len(h_RADAR_SL)-1

#find the index of the minimum element of a 2D array
array3=np.array([[1,200,0.2],[3,0.04,8]])
print("The shape of array3 is:",np.shape(array3))
print('The minimum element of the array3 is',np.amin(array3))
result = np.where(array3 == np.amin(array3))
listOfCordinates = list(zip(result[0], result[1]))
#row_number=listOfCordinates[0]
#col_number=listOfCordinates[1]
row_number=result[0]
col_number=result[1]

print("The row index of the minimum element is:",row_number[0])
print("The col index of the minimum element is:",col_number[0])

Ns=10
L_Tercom = np.zeros((1,Ns))
print("The shape of L_Tercom is:",np.shape(L_Tercom))
L_Tercom[0,1]=10

array4=[[array1[0,0]-array2[1,1]],
        [array1[0,1]-array2[1,1]]]
print('array4:',array4)
print('Shape of array4 is:',np.shape(array4))

array5=np.matmul(array2,array3)
print('shape of array5 is:',np.shape(array5))

array6=np.array([[1],[200],[0.3]])
print('shape of array6 is:',np.shape(array6))
array7=np.zeros([3,10])
print('shape of array7 is:',np.shape(array7))
array7[:,0]=array6[:,0]
print('array7 after assigning first column of array6 to it:\n',array7)
