import numpy as np

### get annotations from txt 
def get_annot(annot_path = 'dataset/LEs920/ann.txt'):
    """get annotation from dataset text"""
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    resdic = {}
    for ann in annotations:
        ann = ann.split(' ')
        image_path = ann[0]
        boxarr = []
        for a in ann[1:]:
            box = np.fromstring(a, sep=',')
            boxarr.append(box)
        boxarr = np.array(boxarr)
        resdic[image_path] = boxarr

    return resdic

### write annotations to a txtfile
def write_annot( anndic, txtfile ):
    """write annoction to txt file
    Parameters: 
    anndic: dictionary, {'image_path.jpg': np.array([[x1,x2,y1,y2,clsid],...])}
    txtfile: ann.txt
    """
    
    imgfiles = list( anndic.keys() )

    with open(txtfile, 'w') as f:
    
        for image_ann in imgfiles:
            bboxes = anndic[image_ann]
            # write boxes to txt file
            for i in range( len(bboxes)):
                x1, y1, x2, y2 = bboxes[i][:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                xmin = str(x1)
                ymin = str(y1)
                xmax = str(x2)
                ymax = str(y2)
                class_ind = str( int( bboxes[i][4] ) )
                image_ann += ' ' + ','.join([xmin, ymin, xmax, ymax, class_ind])
            f.write(image_ann + "\n")
            

### load fits 
def load_fits(image_path, clipfits=True, clipvmax=10, normfits=True, range01=False):
    # change path to fits npz
    fitspath = image_path.replace('images/train/', 'fits/').replace('jpg', 'fits.npz') 

    fitsarr = np.load(fitspath)['fitsarr']
    #image = np.repeat( np.expand_dims(fitarr, axis=2), repeats=3, axis=2)
    if clipfits:
        fitsarr = np.clip(fitsarr, a_min=-clipvmax, a_max=clipvmax)
    if normfits:
        #normalize to mean=0, std=1
        fitsarr = (fitsarr - fitsarr.mean() )/ fitsarr.std()
    if range01:
        # change range to [0, 1]
        fitsarr = (fitsarr + clipvmax) / (clipvmax * 2)
        
    return fitsarr


# plot bounding box
def random_colors(N, bright=True):
    '''
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    '''
    import random
    import colorsys

    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def plot_bboxes(image_path, bboxes, labels=[], ax=None, figsize=(8, 8), readfits=False, title=None,
                colors=None):
    """
    Draw bounding boxes on image
    img: img array
    bboxes: bounding boxes array, shape [n bboxes, 4]
    """
    import cv2
    import random
    import matplotlib.pyplot as plt
    from matplotlib import patches
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
    
    N = bboxes.shape[0]
    if colors == None:
        colors = random_colors(N)
    else: 
        colors = [colors]*N

    if readfits:
        fitspath = image_path.replace("jpg", "fits.npz")
        image_data = np.load(fitspath)["fitsarr"]
        print(fitspath)
        original_image = fits_to_uint8(image_data, vmin=-1, vmax=1)
        ax.imshow(original_image)

    elif image_path:
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)    
        #image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        ax.imshow(original_image)
    
    ax.set_title(title)
    for i in range(N):
        # for faster rcnn
        #y1, x1, y2, x2 = bboxes[i] [:4]
        #y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)

        # for yolo
        x1, y1, x2, y2 = bboxes[i][:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=colors[i], facecolor='none')
        ax.add_patch(p)
        
        # Label
        #class_id = class_ids[i]
        #score = scores[i] if scores is not None else None
        #label = class_names[class_id]
        #x = random.randint(x1, (x1 + x2) // 2)

        if len(labels)!=0:
            caption = "{:.2f}".format( labels[i] )
            ax.text(x2 - 8, y1, caption,
                    color='w', size=10, backgroundcolor="none") 


### draw boxes
def draw_boxes(bboxes, labels=[], ax=None, colors=None):
    """draw bounding box on a given ax, [N, x1, y1, x2, y2]"""
    import random
    import matplotlib.pyplot as plt
    from matplotlib import patches
    
    N = bboxes.shape[0]    
    if colors == None:
        colors = random_colors(N)
    else: 
        colors = [colors]*N

    for i in range(N):
        # for faster rcnn
        #y1, x1, y2, x2 = bboxes[i] [:4]
        #y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)    
        # for yolo
        x1, y1, x2, y2 = bboxes[i][:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=colors[i], facecolor='none')
        ax.add_patch(p)

        # Label
        #class_id = class_ids[i]
        #score = scores[i] if scores is not None else None
        #label = class_names[class_id]
        # x = random.randint(x1, (x1 + x2) // 2)
        if len(labels)!=0:
            caption = "{:.2f}".format( labels[i] )
            ax.text(x2 - 8, y1, caption,
                    color='w', size=10, backgroundcolor="none") 
    

