import os
import matplotlib.pyplot as plt
import numpy as np

root = '/home/user1/ariel/fed_learn/distillation-exps_Moshe/doubledistill/ddistexps/ftdistil/checkpoints_several_temperatures_25epochs_temperature_4/'
folds = [root+fold for fold in os.listdir(root)]
ind = []
model_names = ['resnetdebug', 'resnetsmall', 'resnetlarge']

# Define keys for dictionaries
dictionary_keys = ['val_acc', 't_acc','temperature','shifts', 'cand']

# Create dictionaries and assign to variables
for var_name in model_names:
    globals()[var_name] = {key: [] for key in dictionary_keys}
shifts_names= ['adjust_brightness', 'adjust_contrast','adjust_saturation','adjust_hue','adjust_gamma','autocontrast'\
               ,'invert','solarize','adjust_sharpness','rotate','resize','center_crop','gaussian_blur','pad']

val_transforms_list = [
    'lambda img, brightness_factor=0.5: transforms.functional.adjust_brightness(img, brightness_factor=brightness_factor)',
    'lambda img, contrast_factor=0.5: transforms.functional.adjust_contrast(img, contrast_factor=contrast_factor)',
    'lambda img, saturation_factor=0.5: transforms.functional.adjust_saturation(img, saturation_factor=saturation_factor)',
    'lambda img, hue_factor=0.5: transforms.functional.adjust_hue(img, hue_factor=hue_factor)',
    'lambda img, gamma=0.5: transforms.functional.adjust_gamma(img, gamma=gamma)',
    'lambda img, autocontrast=v2.RandomAutocontrast(): transforms.functional.autocontrast(img)',
    'transforms.functional.invert',
    'partial(transforms.functional.solarize, threshold=1)',
    'partial(transforms.functional.adjust_sharpness, sharpness_factor=0.5)',
    'partial(transforms.functional.rotate, angle=15)',
    'partial(transforms.functional.resize, size=(32, 32))',
    'partial(transforms.functional.center_crop, output_size=(32, 32))',
    'partial(transforms.functional.gaussian_blur, kernel_size=3)',
    'partial(transforms.functional.pad, padding=4)']

# created dictionary for each ran model
temp_val = 4.0
for fold in folds:
    if 'resnetdebug' in fold:
        pts = os.listdir(fold)
        index = fold.split('_ind')[-1]
        shift = [transformed for transformed in shifts_names if transformed in val_transforms_list[int(index)]][0]
        cand = fold.split('student_')[1][0]
        #temperature = float(max([acc.split('-')[5].split('_')[4][:-3] for acc in pts]))
        temperature = [acc.split('-')[5].split('_')[4][:-3] for acc in pts]
        indices = [ind for ind, value in enumerate(temperature) if value == str(temp_val)]
        #temperature = "{:.2f}".format(temperature)
        if indices == []:
            val_acc = 5.0
            t_acc = 80.0
        else:

            temperature = temp_val
            selected_pts = [pts[i] for i in indices]

            val_acc = float(max([acc.split('-')[4].split('_')[:3][-1] for acc in selected_pts]))

            val_acc = "{:.1f}".format(val_acc)
            t_acc = float(max([acc.split('-')[5].split('_')[2] for acc in selected_pts]))
            t_acc = "{:.1f}".format(t_acc)

        resnetdebug['shifts'].append(shift)
        resnetdebug['val_acc'].append(val_acc)
        resnetdebug['t_acc'].append(t_acc)
        resnetdebug['temperature'].append(temperature)
        resnetdebug['cand'].append(cand)
    elif 'resnetsmall' in fold:
        pts = os.listdir(fold)
        index = fold.split('_ind')[-1]
        shift = [transformed for transformed in shifts_names if transformed in val_transforms_list[int(index)]][0]
        cand = fold.split('student_')[1][0]
        #temperature = float(max([acc.split('-')[5].split('_')[4][:-3] for acc in pts]))
        temperature = [acc.split('-')[5].split('_')[4][:-3] for acc in pts]
        indices = [ind for ind, value in enumerate(temperature) if value == str(temp_val)]
        #temperature = "{:.2f}".format(temperature)
        if indices == []:
            val_acc = 5.0
            t_acc = 80.0
        else:

            temperature = temp_val
            selected_pts = [pts[i] for i in indices]

            val_acc = float(max([acc.split('-')[4].split('_')[:3][-1] for acc in selected_pts]))

            val_acc = "{:.1f}".format(val_acc)
            t_acc = float(max([acc.split('-')[5].split('_')[2] for acc in selected_pts]))
            t_acc = "{:.1f}".format(t_acc)

        resnetsmall['shifts'].append(shift)
        resnetsmall['val_acc'].append(val_acc)
        resnetsmall['t_acc'].append(t_acc)
        resnetsmall['temperature'].append(temperature)
        resnetsmall['cand'].append(cand)
    else:
        pts = os.listdir(fold)
        index = fold.split('_ind')[-1]
        shift = [transformed for transformed in shifts_names if transformed in val_transforms_list[int(index)]][0]
        cand = fold.split('student_')[1][0]
        #temperature = float(max([acc.split('-')[5].split('_')[4][:-3] for acc in pts]))
        temperature = [acc.split('-')[5].split('_')[4][:-3] for acc in pts]
        indices = [ind for ind, value in enumerate(temperature) if value == str(temp_val)]
        #temperature = "{:.2f}".format(temperature)
        if indices == []:
            val_acc = 5.0
            t_acc = 80.0
        else:

            temperature = temp_val
            selected_pts = [pts[i] for i in indices]

            val_acc = float(max([acc.split('-')[4].split('_')[:3][-1] for acc in selected_pts]))

            val_acc = "{:.1f}".format(val_acc)
            t_acc = float(max([acc.split('-')[5].split('_')[2] for acc in selected_pts]))
            t_acc = "{:.1f}".format(t_acc)

        resnetlarge['shifts'].append(shift)
        resnetlarge['val_acc'].append(val_acc)
        resnetlarge['t_acc'].append(t_acc)
        resnetlarge['temperature'].append(temperature)
        resnetlarge['cand'].append(cand)

temp_range = [1,2,4]
def plot_fig(resnetdebug,resnetsmall, resnetlarge, exp_name):#, temp_range):


    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot([float(num) for num in resnetsmall['val_acc']],marker ='.', markersize=9,color='b',label='resnetsmall-student_val',
             linewidth=0)
    plt.plot([float(num) for num in resnetdebug['val_acc']],marker ='.',markersize=9, color='r',label='resnetdebug-student_val',
             linewidth=0)
    plt.plot([float(num) for num in resnetlarge['val_acc']],marker ='.',markersize=9, color='g',label='resnetlarge-student_val',
             linewidth=0)
    plt.plot([float(num) for num in resnetsmall['t_acc']], marker='.', markersize=9, color='black', label='teacher-clip_val',
             linewidth=0)
    plt.legend(loc='upper right')
    # Set x-axis labels and rotation for better readability
    plt.xticks(range(len(resnetsmall['shifts'])), resnetsmall['shifts'], rotation=90)  # Rotate x-axis labels for long strings

    # Set axis labels and title
    plt.xlabel("Shifts", fontsize=6)
    plt.ylabel("val_acc")
    plt.ylim(5, 85)
    plt.yticks(range(5, 86, 5))
    plt.grid(True)  # Add grid lines for better visualization
    plt.title(f'Shifts after {exp_name}')
    plt.tight_layout()  # Adjust spacing between elements
    plt.savefig('accuracies_'+exp_name+'.png')
    plt.show()
#use the dictionaries to plot accucaries
exp_name = '25_epochs_temperature_range_1_4_'+str(temp_val)
plot_fig(resnetdebug,resnetsmall, resnetlarge,exp_name)
