from utils import *

def userPreferences(dataset, x_ord):

    x_org = ord2org(x_ord, dataset)

    print('\n')
    print('----- possible values -----')
    for f_val in dataset['feature_values']:
        print(f_val)

    print('\n')
    print('----- instance values -----')
    for i, f in enumerate(dataset['feature_names']):
        print(f+':', x_org[i])

    ## discrete constraints = {'fix', {v1, v2, v3, ...}}
    ## continuous constraints = {'fix', 'l', 'g', 'le', 'ge', [lb, ub]}
    ## constraints = {feature_name_1: (constraint, importance), feature_name_2: (constraint, importance), ...}

    ## Adult data set
    if dataset['name'] == 'adult':
       
        print('\n')
        print('----- user-specified constraints -----')
        constraints = {'age': ('ge',1),
                       'gender': ('fix', 1),
                       'race': ('fix', 1)}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')


    ## Credit card default data set
    elif dataset['name'] == 'credit-card-default':
        
        print('\n')
        print('----- user-specified constraints -----')
        constraints = {'AGE': ('ge', 1),
                       'SEX': ('fix', 1)}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')
         

    ## Student data set
    elif dataset['name'] == 'student-por':
       
        print('\n')
        print('----- user-specified constraints -----')
        constraints = {'age': ('ge', 1),
                       'sex': ('fix', 1),
                       'G1': ([0, 21], 1),
                       'G2': ([0, 21], 1)}

        constraint = [None] * len(x_ord)
        importance = [None] * len(x_ord)
        for p in constraints:
            index = dataset['feature_names'].index(p)
            constraint[index] = constraints[p][0]
            importance[index] = constraints[p][1]
            print(p + ':', constraints[p][0], 'with importance', '(' + str(constraints[p][1]) + ')')            

    print('\n')
    print('N.B. preferences are taken into account when ACTIONABILITY=True!')
    print('\n')

    preferences = {'constraint': constraint,
                   'importance': importance}

    return preferences