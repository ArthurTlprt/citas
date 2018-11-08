logo = '''@&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@#///#/@@@@@@@###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@(//////(%#*..,/./%@@@@@@@@@@@@&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&@@@@@@
@@@&////*(*.#*(,.(.,(#@@@@@@&(//////(@@(//@@(/////////@@@@@//#@@@@@@@%///////@@
@@@@@////(*/,/(../.(*#@@@@@@//%@@@(@@@@(//@@@@@@//(@@@@@@&////@@@@@@#//@@@@@@@@
@@@@@////#./.(.(*,**.#@@@@@#//@@@@@@@@@(//@@@@@@//#@@@@@@///%//@@@@@(//#@&@@@@@
@@@@@@////(,**,*.(/.#%@@@@@#//@@@@@@@@@(//@@@@@@//#@@@@@#//@@//#@@@@@///////((@
@@@@@@/////((,...,#%@@@@@@@#//@@@@@@@@@#//@@@@@@//#@@@@@//%##%//@@@@@@@@@@@///@
@@@@@@@/////////(/@@@@@@@@@#//#&@&%%@@@#//@@@@@@//#@@@@//((((((//#@@//%&@@@///@
@@@@@@@@/////////@@@@@@@@@@@@(/////(@@@#//@@@@@@//#@@@#/(@@@@@@(//@@#///////%@@
@@@@@@@@@@@@@@@(/@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n'''

print(logo)


LorMorA = input('Do you want to perform a leaf analysis? (type l)\nOr do yon want to analyse a microscopic image? (type m)\nOr do you want a lamp assay? (type a)\n')


if LorMorA == 'l':
    # call the model
    model = None
elif LorMorA == 'm':
    model = input('Which model do you want to use?\nBasic classifier (type 1)\nDeep learning (type 2)\n')
    if model == '1':
        from CITASbasic import basic_classifier
        basic_classifier()
    elif model == '2':
        from predict import deep_learning_prediction
        images_path, p = deep_learning_prediction('test_images')
        print('\n\n')
        print('Predictions: ')
        for i, image in enumerate(images_path):
            if p[i][0] >= p[i][1]:
                output = '{} is detected as clean at {} %'.format(image, p[i][0]*100)
                print(output)
            else:
                output = '{} is detected as infected at {} %'.format(image, p[i][1]*100)
                print(output)
elif LorMorA == 'a':
    # call model
    from CITASLampTALAB import lamp_assay
    lamp_assay()
print(LorMorA)
