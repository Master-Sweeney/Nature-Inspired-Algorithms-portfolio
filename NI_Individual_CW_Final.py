#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:26:44 2019

@author: alanngungi
"""
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean


#Here open_file as a variable is used to open the file containing the bag attributes of weight and value
open_file = open("/Users/alanngungi/Desktop/School Work/Nature Inspired Computation/Assessments/Individual /BankProblem.txt")


weight_arr = [] #Array used to hold the weights of the bags
value_arr = []#Array used to hold the values of the bags
Solutions_and_Fitness_values = [] #Array used to hold the fitnesses of the solutions in the population 
float_weight = []#Array used to hold the weight values of the bags after converting them into floating point numbers instead of strings
int_value = []#Array used to hold the values of the bags after converting them into integer numbers instead of strings
Population = []#Array used to hold the Population elements


""" This block turns all the weight and value elements into an array by finding 
the word in the textfile and adding its respective value to the appropriate list """

# string to search in file
word1 = 'weight'
word2 = 'value'
# read all lines in a list
lines = open_file .readlines()
for line in lines:
        # check if the string of word1 is present on a current line
        if line.find(word1) != -1:
            weight_arr.append(line[10:13])
        # otherwise check if the string of word2 is present on a current line
        elif line.find(word2) != -1:           
            value_arr.append(line[9:12])

# Here the weight elements from the textfile are converted into numerical values
for x in weight_arr:
    b = float(x)
    float_weight.append(b)

# Here the value elements from the textfile are converted into numerical values 
for x in value_arr:
    b = int(x)
    int_value.append(b)


#This is a function that determines the weight of an input solution 
def weightfind(solution):
    weight = []
    for i in range(0,len(solution)):
        if solution[i] > 0:
            weight.append(float_weight[i])
            
    weightings = sum(weight)
        
    return weightings


#This is a function that removes bags from an input solution 
def removebags(solution, bags):
    for i in range(0, bags-1):
        #if int(solution[-i]) != 0:
            del solution[-i]
            
    return solution



#Function to generate Solutions encoded using binary encoding 
def Solution_generator(list1, weights, values):
    Solution_generator_list = []
    Solution_generator_list_weights = []
    
    '''This multiplies every weight and value element of the bag txt 
    file with the randomly generated binary number such that 
    random unique solutions are output. '''
    
    for i in range(0, 100):
        v1 = list1[i] * weights[i]
        v2 = list1[i] * values[i]
        Solution_generator_list_weights .append(v1)
        weight_limit = sum(Solution_generator_list_weights)
        if weight_limit < 285.0:#Here we stop the function from producing solutions over weight limit 285
            Solution_generator_list .append(v2)
        else:
            break
 
    return Solution_generator_list  

#Population generator function
def Population_Gen(list1, list2, size):
    #This loop decides how many solutions make up the initial population
    for i in range(0, size):
        random_selection = random.choice([0, 1], size=(100))
        x = Solution_generator(random_selection, list1, list2)
        Population.append(x)
        if len(x)<100:
            df = 100 - len(x)
            nn = np.zeros((df,), dtype=int)
            nn9 = list(nn)
            x = x+nn9
            
        Population.append(x)

    return Population

#Function that is used to determine the fitness of an individual solution

def Individual_Fitness(list1): 
    Invidual_fitness_list = []
    for i in list1:
        Invidual_fitness_list.append(i)
        Fitness = sum(Invidual_fitness_list)
    return Fitness


""" Fitness function used to determine the fitnesses of each solution in the initial population.
    Since all of the solutions produced are below the weight limit, we are only using the 
    value of the bags to measure fitness of a solution."""
 
def Fitness_F(list1):
    fitness_list = []
    for i in range(0, len(list1)-1):
        fitness = Individual_Fitness(list1[i])
        fitness_list.append(fitness)
        
    return fitness_list




''' Tournament selection function that selects t random solutions from the initial population
    and returns the fittest among them
'''
def Tournament_Selection(Pop, t, k):
    Tournament_array = []
    for i in range(0,t):
        random_population_solution = random.randint(k) 
        ''' '100 is chosen here because 100 as an integer was also chosen for generating 
        the initial population. So it's choosing a random solution out of these 100 in 
        the population. '''
        Tornament_member = Pop[random_population_solution]
        Tournament_array.append(Tornament_member)
    
    
    return max(Tournament_array)


''' Crossover function that takes the 2 Parents list from the tournament selection
and a crossover point and then returns the 2 children, as the result of crossing over 
the parents genes.
'''

def SP_Crossover(Two_Parent_List, CP, Do ):
    #CP here refers to the crossover point i.e. the point in the solution that you want to crossover after
    #Do is to know if Crossover is included or not
    if Do == "yes":
        cross_child = []
        Parent_1a = Two_Parent_List[0][:CP+1]
        Parent_2a = Two_Parent_List[1][:CP+1]
        Parent_1b = Two_Parent_List[0][CP+1:]
        Parent_2b = Two_Parent_List[1][CP+1:]
        cross_child_1 = Parent_1a+Parent_2b 
        cross_child_2 = Parent_2a+Parent_1b
        cross_child.append(cross_child_1)
        cross_child.append(cross_child_2)
    elif Do == "no":
        cross_child = Two_Parent_List
    return cross_child
        


''' Mutation function that takes the array of crossed over children and an integer to determine
mutations per chromosome and returns the list mutated by adding or removing bags. For example,
if we mutate the first child (which has bag 92 included) we will get back the first child with
bag 92 excluded and vice-versa.
'''

def MP_Mutation(Two_cross_list, Mutation_per_Chromosome):    
    for i in range(0, Mutation_per_Chromosome):
        size = np.size(Two_cross_list)
        random_gene = random.randint(size)
        if float(Two_cross_list[random_gene]) == 0.0:
            Two_cross_list[random_gene] = int_value[random_gene]
        elif float(Two_cross_list[random_gene]) != 0.0:
            Two_cross_list[random_gene] = 0
            
    return Two_cross_list


'''Further experiment to test the functionality of the Evolutionary Algorithm using a variation
of the conventional Mutation parameter'''

def AD_MP_Mutation(Two_cross_list, Mutation_rate_high, Mutation_rate_low, average_fitness): 
    '''If the individual fitness of the solution is lower than the average fitness of the Population,
    the solution is low quality and thus the mutation rate used is kept high to Mutation_rate_high
    and vice-versa. '''
    
    if Individual_Fitness(Two_cross_list) < average_fitness:
        Mutation_per_Chromosome = Mutation_rate_high
        
    elif Individual_Fitness(Two_cross_list) == average_fitness:
        Mutation_per_Chromosome = Mutation_rate_high
    
    else:
        Mutation_per_Chromosome = Mutation_rate_low
        
    for i in range(0, Mutation_per_Chromosome):
        size = np.size(Two_cross_list)
        random_gene = random.randint(size)
        if float(Two_cross_list[random_gene]) == 0.0:
            Two_cross_list[random_gene] = int_value[random_gene]
        elif float(Two_cross_list[random_gene]) != 0.0:
            Two_cross_list[random_gene] = 0
            
    return Two_cross_list

'''Worst replacement function, whereby the weakest fitness is  determined and removed if i
ts value is lower than that of the child fitness. Here the weakest fitness position is determined, 
then the children are added and the weakest one is removed from the population. In this case, 
the children are added regardless, however if they are weaker than the other solutions, 
they are subsequently removed from the population.
'''      
def Worst_Replacement(Population, Initial_Fitnesses, child1, child2, Weakest):
    Weakest = Initial_Fitnesses.index(min(Initial_Fitnesses))
    Population.append(child1)
    Population.append(child2)
    Population.pop(Weakest)

    return Population


#This is a function that removes solutions from the population
def remove_solution(solution):
    del solution
            
    return 

''' This function runs the entire evolutionary algorithm and returns the Fittest solution after every
iteration of the EA. This is done for the purposes of analysing the evolutionary algorithim in terms of 
Fittness of solutions over each generation.
'''

Fittest_solution =[]
Crossover_value = []
Population_value = []
Mutation_value = []
 
def EA(itera, P, T, Cr, do, Mu):
     Init_Pop = Population_Gen(float_weight, int_value, P)#Initial Population created
     for i in range(0, itera): 
         iteration_counter = i #Iteration count
         print(f'Iteration: {iteration_counter}')   
         Initial_Fitnesses = Fitness_F(Init_Pop) #Fitness of generation
         p1 = Tournament_Selection(Init_Pop, T, P)
         p2 = Tournament_Selection(Init_Pop, T, P)#Binary tournament selection producing 2 parent solutions
         parents = []
         parents.append(p1)
         parents.append(p2)
         crossed_children = SP_Crossover(parents, Cr, do) #Parent solutions crossed over
         mutated_and_crossed_child1 = MP_Mutation(crossed_children[0], Mu)
         mutated_and_crossed_child2 = MP_Mutation(crossed_children[1], Mu)#Crossed over solutions are mutated
         weakest_index = Initial_Fitnesses.index(min(Initial_Fitnesses))
         Worst_Replacement(Init_Pop, Initial_Fitnesses, mutated_and_crossed_child1, mutated_and_crossed_child2, weakest_index)#Worst replacement of solutions generated
         Solution_weight = weightfind(Init_Pop[i])#Solution weights are calculated and penilised for going over weight limit 
         
         if Solution_weight > 285.0:
             removebags(Init_Pop[i],1)
             
             
         End_of_Trial_Population_Fitness = Fitness_F(Init_Pop)
         Max_end_of_Trial_Population_Fitness = max(End_of_Trial_Population_Fitness)
         Fittest_solution.append(Max_end_of_Trial_Population_Fitness)

         
     return Fittest_solution
 
''' This function runs the entire evolutionary algorithm and returns the Fittest solution after a single
trial of 10,000 fitness evaluations. This is for the purposes of analysing the how the Fittest solutions
vary with variations in the parameters of Population, Tournament selection and Mutation.
'''
def EA_Trials(itera, P, T, Cr, do, Mu):
     Init_Pop = Population_Gen(float_weight, int_value, P)#Initial Population created
     for i in range(0, itera): 
         iteration_counter = i
         print(f'Iteration: {iteration_counter}')   
         Initial_Fitnesses = Fitness_F(Init_Pop)#Fitness of generation
         p1 = Tournament_Selection(Init_Pop, T, P)
         p2 = Tournament_Selection(Init_Pop, T, P)#Binary tournament selection producing 2 parent solutions
         parents = []
         parents.append(p1)
         parents.append(p2)
         crossed_children = SP_Crossover(parents, Cr, do) #Parent solutions crossed over
         mutated_and_crossed_child1 = MP_Mutation(crossed_children[0], Mu)
         mutated_and_crossed_child2 = MP_Mutation(crossed_children[1], Mu)#Crossed over solutions are mutated
         weakest_index = Initial_Fitnesses.index(min(Initial_Fitnesses))
         Worst_Replacement(Init_Pop, Initial_Fitnesses, mutated_and_crossed_child1, mutated_and_crossed_child2, weakest_index)#Worst replacement of solutions generated
         Solution_weight = weightfind(Init_Pop[i])#Solution weights are calculated and penilised for going over weight limit 
         
         if Solution_weight > 285.0:
             removebags(Init_Pop[i],5)
             
             
         End_of_Trial_Population_Fitness = Fitness_F(Init_Pop)
         Max_end_of_Trial_Population_Fitness = max(End_of_Trial_Population_Fitness)
         
     Fittest_solution = Max_end_of_Trial_Population_Fitness
         
     return Fittest_solution
 
''' The functions listed below pertain to the further experiments section of the task whereby
Various processes are run on the script to observe the functionality of the Evolutionary
Algorithm
'''

''' This function is responsible for employing the evolutionary algorithm run using the 
previously established Adaptive mutation
'''
def EA_Adaptive_MUT(itera, P, T, Cr, do, MuH, MuL):
     Init_Pop = Population_Gen(float_weight, int_value, P)#Initial Population created
     for i in range(0, itera): 
         iteration_counter = i
         print(f'Iteration: {iteration_counter}')   
         Initial_Fitnesses = Fitness_F(Init_Pop)#Fitness of generation
         average_fitness = mean(Initial_Fitnesses)
         p1 = Tournament_Selection(Init_Pop, T, P)
         p2 = Tournament_Selection(Init_Pop, T, P)#Binary tournament selection producing 2 parent solutions
         parents = []
         parents.append(p1)
         parents.append(p2)
         crossed_children = SP_Crossover(parents, Cr, do) #Parent solutions crossed over
         mutated_and_crossed_child1 = AD_MP_Mutation(crossed_children[0], MuH, MuL, average_fitness)
         mutated_and_crossed_child2 = AD_MP_Mutation(crossed_children[1], MuH, MuL, average_fitness)#Crossed over solutions are mutated
         weakest_index = Initial_Fitnesses.index(min(Initial_Fitnesses))
         Worst_Replacement(Init_Pop, Initial_Fitnesses, mutated_and_crossed_child1, mutated_and_crossed_child2, weakest_index)#Worst replacement of solutions generated
         Solution_weight = weightfind(Init_Pop[i])#Solution weights are calculated and penilised for going over weight limit 
         
         if Solution_weight > 285.0:
             removebags(Init_Pop[i],1)
             
             
         End_of_Trial_Population_Fitness = Fitness_F(Init_Pop)
         Max_end_of_Trial_Population_Fitness = max(End_of_Trial_Population_Fitness)
         Fittest_solution.append(Max_end_of_Trial_Population_Fitness)

         
     return Fittest_solution
 
'''This function is responsible for running the evolutionary algorithm with the remove
entire solutions experimental function
'''
def EA_remove_solutions(itera, P, T, Cr, do, Mu):
     Init_Pop = Population_Gen(float_weight, int_value, P) #Initial Population created
     for i in range(0, itera): 
         iteration_counter = i
         print(f'Iteration: {iteration_counter}')   
         Initial_Fitnesses = Fitness_F(Init_Pop)#Fitness of generation
         p1 = Tournament_Selection(Init_Pop, T, P)
         p2 = Tournament_Selection(Init_Pop, T, P)#Binary tournament selection producing 2 parent solutions
         parents = []
         parents.append(p1)
         parents.append(p2)
         crossed_children = SP_Crossover(parents, Cr, do) #Parent solutions crossed over
         mutated_and_crossed_child1 = MP_Mutation(crossed_children[0], Mu)
         mutated_and_crossed_child2 = MP_Mutation(crossed_children[1], Mu)#Crossed over solutions are mutated
         weakest_index = Initial_Fitnesses.index(min(Initial_Fitnesses))
         Worst_Replacement(Init_Pop, Initial_Fitnesses, mutated_and_crossed_child1, mutated_and_crossed_child2, weakest_index)#Worst replacement of solutions generated
         Solution_weight = weightfind(Init_Pop[i])#Solution weights are calculated and penilised for going over weight limit 
         
         if Solution_weight > 285.0:
             remove_solution(Init_Pop[i])
             
             
         End_of_Trial_Population_Fitness = Fitness_F(Init_Pop)
         Max_end_of_Trial_Population_Fitness = max(End_of_Trial_Population_Fitness)
         Fittest_solution.append(Max_end_of_Trial_Population_Fitness)

         
     return Fittest_solution

'''This function is responsible for running the evolutionary algorithm with the remove
bags from solutions experimental function
'''
def EA_remove_bags(itera, P, T, Cr, do, Mu, Re):
     Init_Pop = Population_Gen(float_weight, int_value, P)
     for i in range(0, itera): 
         iteration_counter = i
         print(f'Iteration: {iteration_counter}')   
         Initial_Fitnesses = Fitness_F(Init_Pop)#Fitness of generation
         p1 = Tournament_Selection(Init_Pop, T, P)
         p2 = Tournament_Selection(Init_Pop, T, P)#Binary tournament selection producing 2 parent solutions
         parents = []
         parents.append(p1)
         parents.append(p2)
         crossed_children = SP_Crossover(parents, Cr, do) #Parent solutions crossed over
         mutated_and_crossed_child1 = MP_Mutation(crossed_children[0], Mu)
         mutated_and_crossed_child2 = MP_Mutation(crossed_children[1], Mu)#Crossed over solutions are mutated
         weakest_index = Initial_Fitnesses.index(min(Initial_Fitnesses))
         Worst_Replacement(Init_Pop, Initial_Fitnesses, mutated_and_crossed_child1, mutated_and_crossed_child2, weakest_index)#Worst replacement of solutions generated
         Solution_weight = weightfind(Init_Pop[i])#Solution weights are calculated and penilised for going over weight limit 
         
         if Solution_weight > 285.0:
             removebags(Init_Pop[i],Re)
             
             
         End_of_Trial_Population_Fitness = Fitness_F(Init_Pop)
         Max_end_of_Trial_Population_Fitness = max(End_of_Trial_Population_Fitness)
         
     Fittest_solution = Max_end_of_Trial_Population_Fitness
         
     return Fittest_solution

'''From here we have the user selected prompts that let us select which results we want to 
observe
'''

Iterations_or_Trials_Prompt = int(input("For running Fittest Solutions against Iterations/Generations, input [1]\nFor running Fittest Solution after every trial (for Parameter experiments, input [2]\nFurther Experiments [3]: "))

if Iterations_or_Trials_Prompt == 1: 
    '''Here a graph of fitness against iterations/generations number is plotted to understand how the 
    fittest solutions evolve over the course of 10,000 fitness evaluations.'''
    Iterations = int(input("Iterations = "))
    Population_value = int(input("Population value = "))
    Tournament_value = int(input("Tournament value = "))
    Mutation_value = int(input("Mutation value = "))
    Crossover_choice = input("Do you want to include Crossover? Input \"yes\" or \"no\": ")
    
    if Crossover_choice == "yes":
        Crossover_value = int(input("Crossover value = "))
    elif Crossover_choice == "no": 
        pass

    Fittest_Solutions = EA(Iterations, Population_value, Tournament_value, Crossover_value, Crossover_choice, Mutation_value)
    Iterations_axis = range(0, Iterations)
    plt.plot(Iterations_axis, Fittest_Solutions)
    plt.xlabel("Iterations") 
    plt.ylabel("Fittest Solutions")
    plt.show()
    

elif Iterations_or_Trials_Prompt == 2:
    '''Here a graph of the fittest solutions against trial number is plotted to understand how the fittest solutions
    evolve over the course of multiple trials with respect to either the Tournament selection, Mutation 
    or Population parameters. { uncomment code to run it seperately from the other function call}'''
    
    Iterations = int(input("Iterations = "))
    Crossover_choice = input("Do you want to include Crossover? Input \"yes\" or \"no\" : ")
    
    if Crossover_choice == "yes":
        Crossover_value = int(input("Crossover value = "))
    elif Crossover_choice == "no": 
        pass
    
    Parameter_Prompt = input("Which variation parameter are you measuring the effect on Trial Fittest solutions? Input \"Population\", \"Tournament\" or \"Mutation\": ")
    
    if Parameter_Prompt == "Population":
        Tournament_value = int(input("Tournament value = "))
        Mutation_value = int(input("Mutation value = "))
        Fittest_val_array = []
        variation_val_array = []
        Trials = int(input("Trials = "))
        for i in range(0, Trials):
            print(f'trial = {i}')
            variation_value = random.randint(50)
            Fittest_Solutions = EA_Trials(Iterations, variation_value, Tournament_value, Crossover_value, Crossover_choice, Mutation_value)
            Fittest_val_array.append(Fittest_Solutions)
            variation_val_array.append(variation_value)
            
        plt.scatter(variation_val_array, Fittest_val_array)
        plt.ylabel("Fittest Solutions")
        plt.xlabel("Population value")
        
    elif Parameter_Prompt == "Tournament":
        Population_value = int(input("Population value = "))
        Mutation_value = int(input("Mutation value = "))
        Fittest_val_array = []
        variation_val_array = []
        Trials = int(input("Trials = "))
        for i in range(1, Trials):
            print(f'trial = {i}')
            variation_value = random.randint(50)
            Fittest_Solutions = EA_Trials(Iterations, Population_value, variation_value, Crossover_value, Crossover_choice, Mutation_value)
            Fittest_val_array.append(Fittest_Solutions)
            variation_val_array.append(variation_value)
            
        plt.scatter(variation_val_array, Fittest_val_array)
        plt.ylabel("Fittest Solutions")
        plt.xlabel("Tournament value")
        
    elif Parameter_Prompt== "Mutation":
        Population_value = int(input("Population value = "))
        Tournament_value = int(input("Tournament value = "))
        Fittest_val_array = []
        variation_val_array = []
        Trials = int(input("Trials = "))
        for i in range(1, Trials):
            print(f'trial = {i}')
            variation_value = random.randint(50)
            Fittest_Solutions = EA_Trials(Iterations, Population_value, Tournament_value, Crossover_value, Crossover_choice, variation_value)
            Fittest_val_array.append(Fittest_Solutions)
            variation_val_array.append(variation_value)
            
        plt.scatter(variation_val_array, Fittest_val_array)
        plt.ylabel("Fittest Solutions")
        plt.xlabel("Mutation value")
        
elif Iterations_or_Trials_Prompt == 3: 
    Prompt = int(input("Adaptive Mutation [1]\nRemove Solutions [2]\nRemove Bags against Fittest[3] : "))
    if Prompt == 1:    
        '''Here a graph of fitness against iterations/generations number is plotted to understand how the 
        fittest solutions evolve over the course of 10,000 fitness evaluations.'''
        Iterations = int(input("Iterations = "))
        Population_value = int(input("Population value = "))
        Tournament_value = int(input("Tournament value = "))
        Mutation_value1 = int(input("Mutation value high= "))
        Mutation_value2 = int(input("Mutation value low= "))
        Crossover_choice = input("Do you want to include Crossover? Input \"yes\" or \"no\": ")
        
        if Crossover_choice == "yes":
            Crossover_value = int(input("Crossover value = "))
        elif Crossover_choice == "no": 
            pass
    
        Fittest_Solutions = EA_Adaptive_MUT(Iterations, Population_value, Tournament_value, Crossover_value, Crossover_choice, Mutation_value1, Mutation_value2)
        Iterations_axis = range(0, Iterations)
        plt.plot(Iterations_axis, Fittest_Solutions)
        plt.xlabel("Iterations") 
        plt.ylabel("Fittest Solutions")
        plt.show()    
        
    elif Prompt == 2:
        '''Here a graph of fitness against iterations/generations number is plotted to understand how the 
        fittest solutions evolve over the course of 10,000 fitness evaluations.'''
        Iterations = int(input("Iterations = "))
        Population_value = int(input("Population value = "))
        Tournament_value = int(input("Tournament value = "))
        Mutation_value = int(input("Mutation value = "))
        Crossover_choice = input("Do you want to include Crossover? Input \"yes\" or \"no\": ")
        
        if Crossover_choice == "yes":
            Crossover_value = int(input("Crossover value = "))
        elif Crossover_choice == "no": 
            pass

        Fittest_Solutions = EA_remove_solutions(Iterations, Population_value, Tournament_value, Crossover_value, Crossover_choice, Mutation_value)
        Iterations_axis = range(0, Iterations)
        plt.plot(Iterations_axis, Fittest_Solutions)
        plt.xlabel("Iterations") 
        plt.ylabel("Fittest Solutions")
        plt.show()
        
    elif Prompt == 3:
        Iterations = int(input("Iterations = "))
        Population_value = int(input("Population value = "))
        Tournament_value = int(input("Tournament value = "))
        Mutation_value = int(input("Mutation value = "))
        Crossover_choice = input("Do you want to include Crossover? Input \"yes\" or \"no\": ")
        if Crossover_choice == "yes":
            Crossover_value = int(input("Crossover value = "))
        elif Crossover_choice == "no": 
            pass
        Fittest_val_array = []
        variation_val_array = []
        Trials = int(input("Trials = "))
        for i in range(1, Trials):
            print(f'trial = {i}')
            variation_value = random.randint(50)
            Fittest_Solutions = EA_remove_bags(Iterations, Population_value, Tournament_value, Crossover_value, Crossover_choice, Mutation_value, variation_value)
            Fittest_val_array.append(Fittest_Solutions)
            variation_val_array.append(variation_value)
            
        plt.scatter(variation_val_array, Fittest_val_array)
        plt.ylabel("Fittest Solutions")
        plt.xlabel("Remove bag value")
        
plt.show()




