import os
import matplotlib.pyplot as plt
import numpy as np

root = '/Data/federated_learning/distillation-exps_Moshe/doubledistill/ddistexps/ftdistil/checkpoints/'
folds = [root+fold for fold in os.listdir(root)]
ind = []
variable_names = ['resnetdebug', 'resnetsmall', 'resnetlarge', 'teacher']

# Define keys for dictionaries
dictionary_keys = ['val_acc', 'shifts']

# Create dictionaries and assign to variables
for var_name in variable_names:
    globals()[var_name] = {key: [] for key in dictionary_keys}
val_transforms_list = [
    'lambda img, brightness_factor=0.5: transforms.functional.adjust_brightness(img, brightness_factor=brightness_factor)',
    'lambda img, contrast_factor=0.5: transforms.functional.adjust_contrast(img, contrast_factor=contrast_factor)',
    'lambda img, saturation_factor=0.5: transforms.functional.adjust_saturation(img, saturation_factor=saturation_factor)',
    'lambda img, hue_factor=0.5: transforms.functional.adjust_hue(img, hue_factor=hue_factor)',
    'lambda img, gamma=0.5: transforms.functional.adjust_gamma(img, gamma=gamma)',
    'transforms.functional.autocontrast',
    'transforms.functional.equalize',
    'transforms.functional.invert',
    'partial(transforms.functional.posterize, bits=4)',
    'partial(transforms.functional.solarize, threshold=128)',
    'partial(transforms.functional.adjust_sharpness, sharpness_factor=0.5)',
    'partial(transforms.functional.rotate, degrees=15)',
    'partial(transforms.functional.resized_crop, size=(32, 32), scale=(0.8, 1.0))',
    'partial(transforms.functional.resize, size=(32, 32))',
    'partial(transforms.functional.center_crop, size=(32, 32))',
    'partial(transforms.functional.gaussian_blur, kernel_size=3)',
    'partial(transforms.functional.perspective, distortion_scale=0.5)',
    'transforms.functional.to_grayscale',
    'transforms.functional.to_pil_image',
    'transforms.functional.to_tensor',
    'transforms.functional.normalize',
    'partial(transforms.functional.pad, padding=4)']
for fold in folds:
    if 'resnetdebug' in fold:
        pts = os.listdir(fold)
        index = fold.split('_')[-1][-1]
        shift = val_transforms_list[int(index)].split('(')[0].split('.')[-1]
        acc = float(max([acc.split('_')[-1][:-3] for acc in pts]))
        acc = "{:.1f}".format(acc)
        resnetdebug['shifts'].append(shift)
        resnetdebug['val_acc'].append(acc)
    elif 'resnetsmall' in fold:
        pts = os.listdir(fold)
        index = fold.split('_')[-1][-1]
        shift = val_transforms_list[int(index)].split('(')[0].split('.')[-1]
        acc = f"{float(max([acc.split('_')[-1][:-3] for acc in pts])):.1f}"
        resnetsmall['shifts'].append(shift)
        resnetsmall['val_acc'].append(float(acc))
    else:
        pts = os.listdir(fold)
        index = fold.split('_')[-1][-1]
        shift = val_transforms_list[int(index)].split('(')[0].split('.')[-1]
        acc = max([acc.split('_')[-1][:-3] for acc in pts])
        teacher['shifts'].append(shift)
        teacher['val_acc'].append(acc)
print(resnetdebug)

shifts = resnetdebug['shifts']
val_acc = resnetdebug['val_acc']
val_acc =[float(x) for x in val_acc]

# Sample data
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(val_acc)

# Set x-axis labels and rotation for better readability
plt.xticks(range(len(shifts)), shifts, rotation=45)  # Rotate x-axis labels for long strings

# Set axis labels and title
plt.xlabel("Shifts")
plt.ylabel("val_acc")
plt.ylim(0, 100)
plt.yticks(range(0, 101, 10))
plt.grid(True)  # Add grid lines for better visualization
plt.tight_layout()  # Adjust spacing between elements
plt.show()

