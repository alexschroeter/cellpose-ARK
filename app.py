from arkitekt_next import register
from mikro_next import Image

from cellpose import denoise, io

class ModelType(str, Enum):
    cyto3 = "cyto3"
    cyto2 = "cyto2"
    cyto = "cyto"
    nuclei = "nuclei"
    tissuenet = "tissuenet_cp3"
    livecell = "livecell_cp3"
    yeast_phc = "yeast_phc_cp3"
    yeast_bf = "yeast_bf_cp3"
    bact_phase = "bact_phase_cp3"
    bact_fluor = "bact_flour_cp3"
    deepbacs = "deepbacs_cp3"
    cyto2_cp3 = "cyto2_cp3"
    custom = "custom"

class RestoreType(str, Enum):
    denoise_cyto3 = "denoise_cyto3"
    deblur_cyto3 = "deblur_cyto3"
    upsample_cyto3 = "upsample_cyto3"
    denoise_nuclei = "denoise_nuclei"
    deblur_nuclei = "deblur_nuclei"
    upsample_nuclei = "upsample_nuclei"

@register(collections=["denoising","prediction",])
def run_cellpose_denoise_model(
    image=Image,
    gpu: bool = True,
    pretrained_model: False,
    mkldnn: True,
    diam_mean: 30,
    device: None,
    nchan=2, 
    pretrained_model_ortho=None, 
    backbone='default',
    ###
    model_type=ModelType.cyto3,
    restore_type=RestoreType.denoise_cyto3,
    custom_model=None,
    channels,
    channel_axis=0,
    diameter=30, # diameter of objects in pixels
    resample=False,
    cellprob_threshold=0.0,
    flow_threshold=0.4,
    do_3D=False,
    stitch_threshold=0.0,
    cytoplasm_channel=0,
    nuclei_channel=None,
    nuclei_retore=False,
    ) -> Image:

    io.logger_setup() # run this to get printing of progress

    if model == ModelType.custom:
        # ToDo: pass the model from Arkitekt
        raise NotImplementedError("Custom models are not yet supported in this function.")

    model = denoise.CellposeDenoiseModel(
        gpu=True, 
        model_type=model
        restore_type=restore_type,
        chan2_restore=nuclei_retore,
    )

    if nuclei_channel is None:
        nuclei_channel = 0
    channels = [channel_to_segment, nuclei_channel]


    img = image.data.sel(c=0, t=0, z=0).data.compute()

    masks, flows, styles, imgs_dn = model.eval(
        img,
        diameter=diameter,
        channels=channels,
    )

    intermediate_result = xr.DataArray(
        results=[imgs_dn, masks, flows, styles],
        dims=["c", "t", "y", "x"],
    )

    result = from_xarray(
        intermediate_result,
        name="Segmented and Denoised " + image.name,
        origins=[image],
        tags=["denoised", "cellpose", "prediction", "segmented"],
        variety=Image.MASK,
    )

    return result

@register(collections=["segmentation","prediction",])
def predict_segmentation(
    image=Image,
) -> Image:
    pass

@register(collections=["denoise","prediction",])
def predict_denoising(
    image=Image,
) -> Image:
    pass

@register(collections=["segmentation","training",])
def train_cellpose_model(

):