from arkitekt_next import register
from mikro_next.api.schema import Image, File, from_array_like, create_dataset
import xarray as xr
import os
# from typing import Tuple

from cellpose import models, io, train, metrics, utils
import zipfile

@register
def import_cellpose_dataset_human_in_the_loop() -> tuple[list[Image], list[Image], list[Image], list[Image]]:
    """
    Import the Cellpose Human in the Loop dataset into Arkitekt.
    This function downloads the dataset from a given URL, extracts it,
    and loads the training and test data along with their corresponding labels.
    """
    url = "https://drive.google.com/uc?id=1HXpLczf7TPCdI1yZY5KV3EkdWzRrgvhQ"
    utils.download_url_to_file(url, "human_in_the_loop.zip")

    with zipfile.ZipFile("human_in_the_loop.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

    train_data, train_labels, train_filename, test_data, test_labels, test_filename = io.load_train_test_data(
        "human_in_the_loop/train/",
        "human_in_the_loop/test/",
        mask_filter="_seg.npy",
    )

    cellpose_dataset = create_dataset("Cellpose - Human in the Loop",)
    train_dataset = create_dataset("Training Data", parent=cellpose_dataset)
    test_dataset = create_dataset("Test Data", parent=cellpose_dataset)

    a, b, c, d = [], [], [], []
    for i in range(len(train_data)):
        a.append(from_array_like(
            xr.DataArray(train_data[i], dims=("c", "y", "x")),
            name=f"{os.path.basename(train_filename[i])}",
            tags=["cellpose", "human_in_the_loop", "training"],
            dataset=train_dataset,
        ))
        b.append(from_array_like(
            xr.DataArray(train_labels[i], dims=("y", "x")),
            name=f"Labelmask of {os.path.basename(train_filename[i])}",
            tags=["cellpose", "human_in_the_loop", "training", "labels"],
            dataset=train_dataset,
        ))
    for i in range(len(test_data)):
        c.append(from_array_like(
            xr.DataArray(test_data[i], dims=("c", "y", "x")),
            name=f"{os.path.basename(test_filename[i])}",
            tags=["cellpose", "human_in_the_loop", "testing"],
            dataset=test_dataset,
        ))
        d.append(from_array_like(
            xr.DataArray(test_labels[i], dims=("y", "x")),
            name=f"Labelmask of {os.path.basename(test_filename[i])}",
            tags=["cellpose", "human_in_the_loop", "testing", "labels"],
            dataset=test_dataset,
        ))
    return a, b, c, d

@register
def import_cellpose_dataset(
        train_archive: File,
        test_archive: File,
) -> tuple[list[Image], list[Image], list[Image], list[Image]]:
    """
    Download the Dataset from https://www.cellpose.org/dataset
    and add it to Arkitekt, from here this function will import
    the training and test data as well as the labels.
    """
    train_archive.download("train.zip")
    test_archive.download("test.zip")

    with zipfile.ZipFile("train.zip", 'r') as zip_ref:
        zip_ref.extractall("cellpose_SAM_dataset/")
    with zipfile.ZipFile("test.zip", 'r') as zip_ref:
        zip_ref.extractall("cellpose_SAM_dataset/")

    train_data, train_labels, train_filename, test_data, test_labels, test_filename = io.load_train_test_data(
        "cellpose_SAM_dataset/train",
        "cellpose_SAM_dataset/test",
        image_filter="_img",
        mask_filter="_masks",
    )

    cellpose_dataset = create_dataset("Cellpose - SAM",)
    train_dataset = create_dataset("Training Data", parent=cellpose_dataset)
    test_dataset = create_dataset("Test Data", parent=cellpose_dataset)

    a, b, c, d = [], [], [], []
    for i in range(len(train_data)):
        a.append(from_array_like(
            xr.DataArray(train_data[i], dims=("x", "y", "c")),
            name=f"{os.path.basename(train_filename[i])}",
            tags=["cellpose", "SAM", "training"],
            dataset=train_dataset,
        ))
        b.append(from_array_like(
            xr.DataArray(train_labels[i], dims=("x", "y")),
            name=f"Labelmask of {os.path.basename(train_filename[i])}",
            tags=["cellpose", "SAM", "training", "labels"],
            dataset=train_dataset,
        ))
    for i in range(len(test_data)):
        c.append(from_array_like(
            xr.DataArray(test_data[i], dims=("x", "y", "c")),
            name=f"{os.path.basename(test_filename[i])}",
            tags=["cellpose", "SAM", "testing"],
            dataset=test_dataset,
        ))
        d.append(from_array_like(
            xr.DataArray(test_labels[i], dims=("x", "y")),
            name=f"Labelmask of {os.path.basename(test_filename[i])}",
            tags=["cellpose", "SAM", "testing", "labels"],
            dataset=test_dataset,
        ))
    return a, b, c, d

@register(collections=["segmentation","prediction",])
def run_cellpose_SAM(
    image=Image,
    pretrained_model: models.CellposeModel = None,
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
    ) -> models.CellposeModel:

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
