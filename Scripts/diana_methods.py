# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:56:11 2019

@author: Karips
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
def dianaCoordinates(height, width,filepath = "./Model.dat",_2D = True, printCoords = True):
   '''
   Function that returns the coordinates of an element 
   from a Model.dat file produced by Diana.
   ---------------------------------------------------
   Dependencies:
       import os
       import matplotlib.pyplot as plt
   ---------------------------------------------------
   Parameters:
       filepath: str -> The path fof the model file
       _2D: bool -> If model is 2D (only 2D atm)
       height: float -> The height of the element
       width: float -> The width of the element
       printCoords: bool -> Whether to print the scattered data with 
                            preset plot configuration
    --------------------------------------------------
    Returns:
        final_coordinates: tuple -> A tuple with the 2D coordinates
                            of the elements (xx,yy)
   '''
   
   if _2D:
       datContent = [i.strip().split() for i in open(filepath).readlines()]
        
       temp_index = datContent.index(["'COORDINATES'"])
       temp_index2 = datContent.index(["'MATERI'"])
        
       coordinates = datContent[temp_index+1:temp_index2]
       isModel2D = True
       final_coordinates=[]
       if isModel2D: 
           for i in coordinates: i.pop(3)
    
       for i in range(len(coordinates)): 
           coordinates[i].pop(0)
           for j in range(len(coordinates[i])): 
               coordinates[i][j]=float(coordinates[i][j])
           if (abs(coordinates[i][0])<=width and coordinates[i][1]>=0 
               and coordinates[i][1]<=height):
               final_coordinates.append(coordinates[i])
        
       final_coordinates = [*zip(*final_coordinates)]  
       if printCoords:
           plt.scatter(final_coordinates[0],final_coordinates[1],marker='o',s = 1)
       return final_coordinates
   else: 
       print('3D not supported yet')
       return None
	   
def iterate_data(data,line,prints):
    '''
	   Function that iterates and prints through the 
	   list that is generated from the "Results.t" file.
	   ---------------------------------------------------
	   Dependencies:
	   ---------------------------------------------------
	   Parameters:
			data: list -> The list to iterate
			line: int -> Stores the last line to keep iterating
						the rest of the list
			prints: int -> Number of lines to print each iteration
		--------------------------------------------------
		Returns:
			line: int -> THe last printed line
    '''
    for i in range(prints):
        print(data[line])
        line+=1
    return line
	
def insertsion(data,stepLine_length): 
    '''
       Function that inserts an empty string element at the 
       beggining of the elements 2-4, to match the columns
       numbers with the element 1 (contains step)
       NOTE: the function mutates the original list
	   ---------------------------------------------------
	   Dependencies:
	   ---------------------------------------------------
	   Parameters:
			data: list -> The list to modify
            stepLine_length: int -> Necessairy because SXX and EXX are 
            parsed differently
		--------------------------------------------------
		Returns:
    '''
    for i in range(len(data)):
        if data[i][0].isdigit():
            if len(data[i])==stepLine_length:
                data[i].insert(0,' ') 
				
def Diana2D_preparation(filepath = "Results.tb",feature = "EXX",
                        saveResults=None):
    '''
       Function that prepares the 2D beam data for ML prediction
       from a Diana file.
	   ---------------------------------------------------
	   Dependencies:
                  import os
                  import numpy as np
	   ---------------------------------------------------
	   Parameters:
			filepath: str -> Path to file
            feature: str -> Either EXX or SXX (atm) 
			saveResults: str -> name of the file to store values.
                         Values are stored in .npy file.
                         If name not give, results are not saved.
		--------------------------------------------------
		Returns:
            final_values: np.array -> the beam values for all timesteps
    '''
    feature_dictionairy ={"E":5, "S":4}
    num_el = 0
    data_compact = []
    momentWriteFlag = False
    #check if the directory is correct
    print(os.getcwd())   
    #    line = 0 #For the iterate_data function
    #Store each line in a list
    data = [i.strip().split() for i in open(filepath).readlines()]
    i_min = 99999
    #Remove all empty lists (empty lines)
    data = list(filter(lambda a: a != [], data))
    insertsion(data,feature_dictionairy[feature[0]])
    #Keep only (EXX,EYY,EZZ,GXY)||(SXX,SYY,SZZ) and iteration numbers  
    for i in range(len(data)):
        if data[i][0]=='Step':
            if data[i+5][0]=='Elmnr' and data[i+5][2]==feature:
                momentWriteFlag = True
                
        if data[i][0]=='Analysis': 
            momentWriteFlag = False
            if num_el>0: i_min=i
        if momentWriteFlag == True:
            data_compact.append(data[i])
            if data[i][0].isdigit() and len(data[i])==feature_dictionairy[
                    feature[0]]+1 and i<i_min:
                num_el +=1
        
    data_final =[]
    for i in range(len(data_compact)):
        if data_compact[i][0].isdigit():
            data_final.append([float(j) for j in data_compact[i]])
            data_final.append([float(j) for j in data_compact[i+1][1:]])
            data_final.append([float(j) for j in data_compact[i+2][1:]])
            data_final.append([float(j) for j in data_compact[i+3][1:]])
     
    temp_meanVals = []
    final_values = []
    for i in range(0,len(data_final),4):
    #    print(i)
        temp_meanVals.append((data_final[i][2]+data_final[i+1][1]+
                              data_final[i+2][1]+data_final[i+3][1])/4)
        if len(temp_meanVals)==num_el:
            final_values.append(temp_meanVals)
            temp_meanVals=[]
     
    final_values = np.asmatrix(final_values)
    if saveResults is not None:
        np.save(saveResults,final_values)
    return final_values

def Diana3D_SXX_preparation(filepath = "Rebar.tb",
                        saveResults=None, numberOfLongBars = 2):
    '''
       Function that prepares the 3D beam data for ML prediction
       from a Diana file. ONLY SXX
	   ---------------------------------------------------
	   Dependencies:
                  import os
                  import numpy as np
                  import pickle
	   ---------------------------------------------------
	   Parameters:
			filepath: str -> Path to file
			saveResults: str -> name of the file to store values.
                         Values are stored in .npy file.
                         If name not give, results are not saved.
		--------------------------------------------------
		Returns:
            final_values: np.array -> the beam values for all timesteps 
                            for all rebar elements
    '''
    data_compact = []
    #saveResults = 'SXX_bar'
    momentWriteFlag = False
    #check if the directory is correct
    print(os.getcwd())   
    #    line = 0 #For the iterate_data function
    #Store each line in a list
    data = [i.strip().split() for i in open(filepath).readlines()]
    #Remove all empty lists (empty lines)
    data = list(filter(lambda a: a != [], data))
    #Keep only (EXX,EYY,EZZ,GXY)||(SXX,SYY,SZZ) and iteration numbers  
    for i in range(len(data)):
        if data[i][0]=='Step':
            if data[i+5][0]=='Reinr':
                momentWriteFlag = True
                
        if data[i][0]=='Analysis': 
            momentWriteFlag = False
        if momentWriteFlag == True:
            data_compact.append(data[i])
    
    numOfTimesteps = max([int(i[2]) for i in list(filter(lambda a: a[0]=='Step',
                          data_compact))])
        
    data_final =[[] for item in range(numberOfLongBars)]
    
    for i in range(len(data_compact)):
        if data_compact[i][0].isdigit():
            bar = int(data_compact[i][0])
            data_final[bar-1].append([float(j) for j in data_compact[i]][-3])
    
    #Reshape according to timesteps
    
    counter = 0;
    for i in data_final:
        data_final[counter] = np.asarray(i).reshape(
                  numOfTimesteps,len(i)//numOfTimesteps)
        counter+=1
     
    if saveResults is not None:      
        with open(saveResults+".txt", "wb") as fp:
            pickle.dump(data_final,fp)
    return data_final
	
def Diana3D_EXX_preparation(filepath = "Rebar.tb",
                        saveResults=None, numberOfLongBars = 2):
    '''
       Function that prepares the 3D beam data for ML prediction
       from a Diana file. ONLY EXX
	   ---------------------------------------------------
	   Dependencies:
                  import os
                  import numpy as np
                  import pickle
	   ---------------------------------------------------
	   Parameters:
			filepath: str -> Path to file
			saveResults: str -> name of the file to store values.
                         Values are stored in .npy file.
                         If name not give, results are not saved.
		--------------------------------------------------
		Returns:
            final_values: np.array -> the beam values for all timesteps 
                            for all rebar elements
    '''
    data_compact = []
    #saveResults = 'EXX_bar'
    momentWriteFlag = False
    #check if the directory is correct
    print(os.getcwd())   
    #    line = 0 #For the iterate_data function
    #Store each line in a list
    data = [i.strip().split() for i in open(filepath).readlines()]
    #Remove all empty lists (empty lines)
    data = list(filter(lambda a: a != [], data))
    #Keep only (EXX,EYY,EZZ,GXY)||(SXX,SYY,SZZ) and iteration numbers  
    for i in range(len(data)):
        if data[i][0]=='Step':
            if data[i+5][0]=='Reinr':
                momentWriteFlag = True
                
        if data[i][0]=='Analysis': 
            momentWriteFlag = False
        if momentWriteFlag == True:
            data_compact.append(data[i])
    
    numOfTimesteps = max([int(i[2]) for i in list(filter(lambda a: a[0]=='Step',
                          data_compact))])
        
    data_final =[[] for item in range(numberOfLongBars)]
    
    for i in range(len(data_compact)):
        if data_compact[i][0].isdigit():
            bar = int(data_compact[i][0])
            data_final[bar-1].append([float(j) for j in data_compact[i]][-3])
    
    #Reshape according to timesteps
    
    counter = 0;
    for i in data_final:
        data_final[counter] = np.asarray(i).reshape(
                  numOfTimesteps,len(i)//numOfTimesteps)
        counter+=1
     
    if saveResults is not None:      
        with open(saveResults+".txt", "wb") as fp:
            pickle.dump(data_final,fp)
    return data_final