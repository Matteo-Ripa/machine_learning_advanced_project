import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json




directory_img = r'/home/matteo/Desktop/GU/2nd_year/machine_learning_advanced/project_course/CODE/images'
directory_dt = r'/home/matteo/Desktop/GU/2nd_year/machine_learning_advanced/project_course/CODE/data'

data_size= 10000
dist_size= 100
dist_names= ['Beta', 'Chi-squared', 'Exponential', 'Gamma', 'Laplace', 'Normal', 'Uniform', 'Weibul']

def make_data(data_size, dist_names, dist_size, dir_images, dir_data, alpha=1.5, beta=2, df=3, scale=0.5, shape=1, loc=0, a=0, b=1):
    img_idx_iter= 0

    data= {}
    for i in range(0, int(data_size/len(dist_names))):
        print(i)
        # if i%100 == 0:
        #     print(f'Step: {i}/{int(data_size/len(dist_names))}')

        beta_dist= np.random.beta(a= alpha, b= beta, size=dist_size)
        chiSq= np.random.chisquare(df= df, size=dist_size)
        exp= np.random.exponential(scale= scale, size=dist_size)
        gamma= np.random.gamma(shape= alpha, scale= beta, size=dist_size)
        lapla= np.random.laplace(loc= loc, scale= scale, size=dist_size)
        norm=np.random.normal(loc= loc, scale= scale, size=dist_size)
        uni= np.random.uniform(low=a, high=b, size=dist_size)
        weib= scale * np.random.weibull(a= shape, size=dist_size)

        dist_list= [beta_dist, chiSq, exp, gamma, lapla, norm, uni, weib]

        beta_params= {'params_interval': [((0, float('inf')), 'alpha'), ((0, float('inf')), 'beta')],
                      'support_interval': (0, 1),
                      'values_params': [alpha, beta]}

        chiSq_params= {'params_interval': [((0, float('inf')), 'df')],
                        'support_interval': (0, float('inf')),
                        'values_params': [df]}

        exp_params= {'params_interval': [((0, float('inf')), 'scale')],
                        'support_interval': (0, float('inf')),
                        'values_params': [scale]}

        gamma_params= {'params_interval': [((0, float('inf')), 'alpha'), ((0, float('inf')), 'beta')],
                        'support_interval': (0, float('inf')),
                        'values_params': [alpha, beta]}

        lapla_params= {'params_interval': [((float('-inf'), float('inf')), 'location'), ((0, float('inf')), 'scale')],
                        'support_interval': (float('-inf'), float('inf')),
                        'values_params': [loc, scale]}

        norm_params= {'params_interval': [((float('-inf'), float('inf')), 'location'), ((0, float('inf')), 'scale')],
                        'support_interval': (float('-inf'), float('inf')),
                        'values_params': [loc, scale]}

        uni_params= {'params_interval': [((float('-inf'), 'b'), 'a'), (('a', float('inf')), 'b')],
                        'support_interval': ('a', 'b'),
                        'values_params': [a, b]}

        weib_params= {'params_interval': [((0, float('inf')), 'scale'), ((0, float('inf')), 'shape')],
                        'support_interval': (0, float('inf')),
                        'values_params': [scale, shape]}

        params_list= [beta_params, chiSq_params, exp_params, gamma_params, lapla_params, norm_params, uni_params, weib_params]

        for distr, k, dist_name, param in zip(dist_list, range(len(dist_list)), dist_names, params_list):
            # Create image name file
            os.chdir(dir_images)
            idx_img= str(k + img_idx_iter)
            img_name= 'img_' + idx_img + '.png'
            sns.kdeplot(distr, fill=True, color= 'black')
            plt.xlim(-2, 4)
            plt.ylim(0, 2)
            plt.grid(False)
            plt.savefig(img_name, dpi=100, bbox_inches='tight')
            plt.close()  # prevents the plot from showing

            #Dataset
            dic_params= {}
            for interval, para_name in param['params_interval']:
                dic_params[para_name]= interval
            data[img_name]= {'label': dist_name,
                             'parameters': dic_params,
                             'values_parameters': param['values_params'],
                             'support': param['support_interval'],
                             'values_support': distr.tolist()}

        img_idx_iter+=len(dist_names)

    os.chdir(dir_data)
    with open('data.json', 'w') as f:
        json.dump(data, f)

    return data

make_data(data_size, dist_names, dist_size, dir_images= directory_img, dir_data= directory_dt)
