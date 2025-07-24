from arkitekt_next import register
from mikro_next import Image

from cellpose import models, io, train, metrics

@register(collections=["segmentation","prediction",])
def run_cellpose_SAM(
    image=Image,
    pretrained_model: Model = None,
    gpu: bool = True,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    tile_norm_blocksize: int = 0,
    ) -> Image:

    io.logger_setup() # run this to get printing of progress

    model = models.CellposeModel(
        gpu=True,
        pretrained_model=pretrained_model.path if pretrained_model else None,
    )

    # @jhnnsrs how do I pass the channels and dimensions I want to use?
    image.data.sel(c="0", t="0").data.compute()
    
    masks, flows, styles = model.eval(
        img_selected_channels, 
        batch_size=32, 
        flow_threshold=flow_threshold, 
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
    )

    # @jhnnsrs Do we want to also upload the flows and styles?
    array = xr.DataArray(
        results=masks,
        dims=["x", "y"],
    )

    result = from_xarray(
        array,
        name="Segmented and Denoised " + image.name,
        origins=[image],
        tags=["cellpose", "prediction", "segmented"],
        variety=Image.MASK,
    )

    return result

@register(collections=["segmentation","training",])
def train_cellpose_SAM(
    model_name: str = "custom_cellpose_model",
    image=Image,
    n_epochs = 100,
    learning_rate = 1e-5,
    weight_decay = 0.1,
    batch_size = 1,
    test: bool = False,
    ) -> Model:

    # Loading the default model which training is added to
    model = models.CellposeModel(gpu=True)

    # @jhnnsrs How do I load a list of images? And how do I select the label masks for training?
    output = io.load_train_test_data(
        train_dir,
        test_dir,
        mask_filter="_seg.npy",
    )
    # ToDo: Read in the data not from file but from mikro directly
    #       We need our own load_train_test_data function that does this

    # (not passing test data into function to speed up training) 
    # AW: I think this is mostly filenames that get dropped which probably blocks 
    # the train_seg function from loading the data from file again
    train_data, train_labels, _, test_data, test_labels, _ = output

    new_model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        nimg_per_epoch=max(2, len(train_data)), # can change this
        model_name=model_name,    
    )

    ### test the model on the test data

    # run model on test images
    masks = model.eval(test_data, batch_size=32)[0]

    # check performance using ground truth labels
    ap = metrics.average_precision(test_labels, masks)[0]
    print('')
    print(f'>>> average precision at iou threshold 0.5 = {ap[:,0].mean():.3f}')

    # save and upload the model
    archive = shutil.make_archive("active_model", "zip", f"{new_model_path}")
    my_model = create_model(
        "active_model.zip",
        kind=ModelKind.PYTORCH,
        name=f"Cellpose Pretrained Model: {pretrained}",
        contexts=[],
    )
    return my_model
