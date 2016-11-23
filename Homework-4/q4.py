%Memetic Algorithm

import random
import sys
n=8
length_of_population=35
max_iter=100

#1 generation of random population
gene_population=[]

def random_population_generation():
    global n
    global length_of_gene_population
    for i in range(length_of_gene_population):
        random_arr=[]
        random_arr=range(1, n+1)
        random.shuffle(random_arr)
        gene_population.append(random_arr)
        
    print len(gene_population)    

def swap(x,i):
    a,b=x[i],x[i+1]
    x[i]=b
    x[i+1]=a
    return x
	
#if score of fitness function is 0 then it is solution, the queen is not intersecting with any other queen


def local_search(x):
    global n
    #print "before local search x= ",x   
    for i in range(n-1):
        x_swap=swap(x,i)
        x_fscore=fitness_score(x)
        xswap_fscore=fitness_score(x_swap)
        if(x_fscore==0):
            print "solution is found: ",x
            quit
            sys.exit() 
        if(xswap_fscore==0):
            print "solution is found: ",x_swap
            quit
            sys.exit() 
        if (x_fscore<xswap_fscore):
            x=x_swap
       
   
    return x

def mutate(x):
  
    i,j=random.sample(range(len(x)), 2)
    a,b=x[i],x[j]
    x[i]=b
    x[j]=a
    return x
    
def doing_search_locally():
    iter=0
    for x in  gene_population:
        gene_population[iter]=local_search(x)
        iter=iter+1

def score_fitness_function(x):
    
    fscore=0
    for i in range(len(x)):
        cntr=0
        for j in range(i+1,len(x)):
            cntr=cntr+1
            if(abs(x[i]-x[j])==cntr):
                fscore=fscore+1
    return fscore

		
random_population_generation()
#local searcing for hill climbing
doing_search_locally()

#sorting the solution according to fitness function

iter=0
score_fitness_function_vals=[]
for x in  gene_population:
    score_fitness_function_vals.append(score_fitness_function(x))
    iter=iter+1 
    
    
#sorting gene_population according to fitness score 
gene_population=[a for (b,a) in sorted(zip(score_fitness_function_vals,gene_population))]

 
if(score_fitness_function(gene_population[0])==0):
    print "solution is found ", gene_population[0]
    quit    
    

#removing old population and generating the new population	
#Generation of new gene_population

#removing last 1/3 population
removing_index=int(length_of_gene_population*0.66)
for i in range(removing_index,length_of_gene_population):
    gene_population.pop(removing_index)

#making new 1/3 of gene_population by mutating with best 1/3 of gene_population  
regeneration_index=int(length_of_gene_population*0.33)


for i in range(regeneration_index+2):
    gene_population.append(mutate(gene_population[i]))



for i in range(max_iter):
    doing_search_locally()
    
    #calculating fitness score
    iter=0
    score_fitness_function_vals=[]
    for x in  gene_population:
        score_fitness_function_vals.append(score_fitness_function(x))
        iter=iter+1 
        
        
    #sorting gene_population according to fitness score 
    gene_population=[a for (b,a) in sorted(zip(score_fitness_function_vals,gene_population))]
    #gene_population.sort(key=lambda p:p[8])
     
    if(score_fitness_function(gene_population[0])==0):
        print "solution is found ", gene_population[0]
        sys.exit()    
     
    #removing LAST 1/3rd
    removing_index=int(length_of_gene_population*0.66)
    for i in range(removing_index,length_of_gene_population):
        gene_population.pop(removing_index)
    
    #regenerating last 1/3 of gene_population by mutating with best 1/3 of gene_population  
    regeneration_index=int(length_of_gene_population*0.33)
    
    for i in range(regeneration_index+2):        
        gene_population.append(mutate(gene_population[i]))
   
    
    
    
    
