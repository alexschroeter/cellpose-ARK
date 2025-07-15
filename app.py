from arkitekt_next import register

from cellpose import denoise, io


@register(collections=["segmentation","prediction",])
def run_segmentation():
    # ToDo: Segmentation is basically the same but you don't really want to run it twice
    #       So it would be nice if you could choose what of the results you want.
    #       i.e denoising or segmentation or both ... Not sure what the best solution is here.
    pass

@register(collections=["denoising","prediction",])
def run_denoising(imgs=None):

    # ToDo: Load the data
    # dat = np.load(filename, allow_pickle=True)["arr_0"].item()
    
    if imgs is None:
        print("Error: No images provided for denoising")
        return None

    io.logger_setup() # run this to get printing of progress

    # DEFINE CELLPOSE MODEL
    # model_type="cyto3" or "nuclei", or other model
    # restore_type: "denoise_cyto3", "deblur_cyto3", "upsample_cyto3", "denoise_nuclei", "deblur_nuclei", "upsample_nuclei"
    model = denoise.CellposeDenoiseModel(
        gpu=True, 
        model_type="cyto3",
        restore_type="denoise_cyto3",
        )

    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    # channels = [0,0]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus
    # OR if you have different types of channels in each image
    # channels = [[2,3], [0,0], [0,0]]

    # if you have a nuclear channel, you can use the nuclei restore model on the nuclear channel with
    # model = denoise.CellposeDenoiseModel(..., chan2_restore=True)

    # NEED TO SPECIFY DIAMETER OF OBJECTS
    # in this case we have them from the ground-truth masks
    diams = dat["diam_test"]

    masks, flows, styles, imgs_dn = model.eval(imgs, diameter=diams, channels=[0,0])

    ### Return the denoised images

    # plt.figure(figsize=(8,12))
    # for i, iex in enumerate([2, 18, 20]):
    #     img = imgs[iex].squeeze()
    #     plt.subplot(3,3,1+i)
    #     plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    #     plt.axis('off')
    #     plt.title("noisy")

    #     img_dn = imgs_dn[iex].squeeze()
    #     plt.subplot(3,3,4+i)
    #     plt.imshow(img_dn, cmap="gray", vmin=0, vmax=1)
    #     plt.axis('off')
    #     plt.title("denoised")

    #     plt.subplot(3,3,7+i)
    #     plt.imshow(img_dn, cmap="gray", vmin=0, vmax=1)
    #     outlines = utils.outlines_list(masks[iex])
    #     for o in outlines:
    #         plt.plot(o[:,0], o[:,1], color=[1,1,0])
    #     plt.axis('off')
    #     plt.title("segmentation")

    # plt.tight_layout()
    # plt.show()
