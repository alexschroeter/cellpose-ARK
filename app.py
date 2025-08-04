from arkitekt_next import register
import xarray as xr
import numpy as np
import os
from mikro_next.api.schema import (
    Image,
    File, 
    from_array_like, 
    create_dataset, 
    PartialInstanceMaskViewInput, 
    create_reference_view,
)
from kraph.api.schema import (
    GraphQuery,
    Model,
    Pairs,
    ViewKind,
    create_graph,
    create_graph_query,
    create_model,
    create_structure,
    create_structure_relation,
    create_structure_relation_category,
)
from typing import cast, Generator, Tuple
from cellpose import models, io, train, metrics, utils
import zipfile

from asyncio import graph

# @context
# @dataclass
# class CurrentModel:
#     model_id: str
#     cellpose_model: object

@register
def import_cellpose_dataset_human_in_the_loop() -> tuple[list[Image], list[Image], list[Image], list[Image]]:
    """
    Import the Cellpose Human in the Loop dataset into Arkitekt.
    This function downloads the dataset from the authors google drive, extracts it,
    and loads the training and test data along with their corresponding labels.
    """
    url = "https://drive.google.com/uc?id=1HXpLczf7TPCdI1yZY5KV3EkdWzRrgvhQ"
    utils.download_url_to_file(url, "datasets/human_in_the_loop.zip")

    with zipfile.ZipFile("datasets/human_in_the_loop.zip", 'r') as zip_ref:
        zip_ref.extractall("datasets/")

    train_data, train_labels, train_filename, test_data, test_labels, test_filename = io.load_train_test_data(
        "datasets/human_in_the_loop/train/",
        "datasets/human_in_the_loop/test/",
        mask_filter="_seg.npy",
    )

    cellpose_dataset = create_dataset("Cellpose - Human in the Loop",)
    train_dataset = create_dataset("Training Data", parent=cellpose_dataset)
    test_dataset = create_dataset("Test Data", parent=cellpose_dataset)

    a, b, c, d = [], [], [], []
    graph = create_graph("Cellpose", description="The default cellpose graph")

    for i in range(len(train_data)):
        a.append(from_array_like(
            xr.DataArray(train_data[i], dims=("c", "x", "y")),
            name=f"{os.path.basename(train_filename[i])}",
            tags=["cellpose", "human_in_the_loop", "training"],
            dataset=train_dataset,
        ))
        b.append(from_array_like(
            xr.DataArray(train_labels[i], dims=("x", "y")),
            name=f"Labelmask of {os.path.basename(train_filename[i])}",
            tags=["cellpose", "human_in_the_loop", "training", "labels"],
            dataset=train_dataset,
            instance_mask_views=[
                PartialInstanceMaskViewInput(
                    referenceView=create_reference_view(
                        image=a[-1],
                        c_min=0,
                        c_max=1,
                    ),
                ),
                ],
        ))

        IS_MASK_FOR = create_structure_relation_category(
            graph,
            "is_mask_for",
            source_definition=b[-1],
            target_definition=a[-1],
            description="This relation connects images with their corresponding masks.",
        )

        b[-1] | IS_MASK_FOR() | a[-1]

    for i in range(len(test_data)):
        c.append(from_array_like(
            xr.DataArray(test_data[i], dims=("c", "x", "y")),
            name=f"{os.path.basename(test_filename[i])}",
            tags=["cellpose", "human_in_the_loop", "testing"],
            dataset=test_dataset,
        ))
        d.append(from_array_like(
            xr.DataArray(test_labels[i], dims=("x", "y")),
            name=f"Labelmask of {os.path.basename(test_filename[i])}",
            tags=["cellpose", "human_in_the_loop", "testing", "labels"],
            dataset=test_dataset,
            instance_mask_views=[
                PartialInstanceMaskViewInput(
                    referenceView=create_reference_view(
                        image=c[-1],
                        c_min=0,
                        c_max=1,
                    ),
                ),
                ],
        ))
        
        IS_MASK_FOR = create_structure_relation_category(
            graph,
            "is_mask_for",
            source_definition=d[-1],
            target_definition=c[-1],
            description="This relation connects images with their corresponding masks.",
        )

        d[-1] | IS_MASK_FOR() | c[-1]

    return a, b, c, d

@register
def import_cellpose_dataset(
        train_archive: File,
        test_archive: File,
) -> tuple[list[Image], list[Image], list[Image], list[Image]]:
    """
    After you have downloaded the Cellpose SAM dataset 
    from https://www.cellpose.org/dataset, accepted the EULA,
    and added it to Arkitekt, from here this function will import
    the training and test data as well as the labels.
    """
    train_archive.download("datasets/train.zip")
    test_archive.download("datasets/test.zip")

    with zipfile.ZipFile("datasets/train.zip", 'r') as zip_ref:
        zip_ref.extractall("datasets/cellpose_SAM/")
    with zipfile.ZipFile("datasets/test.zip", 'r') as zip_ref:
        zip_ref.extractall("datasets/cellpose_SAM/")

    train_data, train_labels, train_filename, test_data, test_labels, test_filename = io.load_train_test_data(
        "datasets/cellpose_SAM/train",
        "datasets/cellpose_SAM/test",
        image_filter="_img",
        mask_filter="_masks",
    )

    cellpose_dataset = create_dataset("Cellpose - SAM",)
    train_dataset = create_dataset("Training Data", parent=cellpose_dataset)
    test_dataset = create_dataset("Test Data", parent=cellpose_dataset)

    a, b, c, d = [], [], [], []
    graph = create_graph("Cellpose", description="The default cellpose graph")

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
            mask_views=[]
        ))
        
        IS_MASK_FOR = create_structure_relation_category(
            graph,
            "is_mask_for",
            source_definition=b[-1],
            target_definition=a[-1],
            description="This relation connects images with their corresponding masks.",
        )

        b[-1] | IS_MASK_FOR() | a[-1]
        
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
        
        IS_MASK_FOR = create_structure_relation_category(
            graph,
            "is_mask_for",
            source_definition=b[-1],
            target_definition=a[-1],
            description="This relation connects images with their corresponding masks.",
        )

        d[-1] | IS_MASK_FOR() | c[-1]
        
    return a, b, c, d

@register(collections=["segmentation","prediction",])
def run_cellpose_SAM(
    image: Image,
    pretrained_model: File = None,
    gpu: bool = True,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    tile_norm_blocksize: int = 0,
    ) -> tuple[Image, Image, Image]:

    io.logger_setup() # run this to get printing of progress

    model = models.CellposeModel(
        gpu=gpu,
        pretrained_model=pretrained_model.path if pretrained_model else "cpsam",
    )

    # @jhnnsrs how do I pass the channels and dimensions I want to use?
    data = image.data.data.compute()
    print(f"Original shape: {data.shape}")
    
    # Transform from (c,t,z,y,x) to (y,x,c) and ensure 3 channels
    # Remove singleton dimensions t and z
    data = data.squeeze(axis=(1, 2))  # Now (c, y, x)
    # Transpose to (y, x, c)
    data = data.transpose(1, 2, 0)  # Now (y, x, c)
    
    # If we don't have 3 channels, pad or duplicate to get 3
    if data.shape[2] < 3:
        # Duplicate the first channel to make it 3 channels
        missing_channels = 3 - data.shape[2]
        if missing_channels == 1:
            # Add first channel again
            data = np.concatenate([data, data[:, :, :1]], axis=2)
        elif missing_channels == 2:
            # Add first channel twice
            data = np.concatenate([data, data[:, :, :1], data[:, :, :1]], axis=2)
    elif data.shape[2] > 3:
        # Take only first 3 channels
        data = data[:, :, :3]
    
    print(f"Transformed shape: {data.shape}")

    ### Craete Mikro-Next Relation
    masks, flows, styles = model.eval(
        data,
        # do_3D=True,
        batch_size=32, 
        flow_threshold=flow_threshold, 
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
    )
    # print(masks[0])
    mask = from_array_like(
            xr.DataArray(masks, dims=("x", "y")),
            name=f"Predicted mask of {image.name}",
            tags=["cellpose", "SAM", "prediction"],
            instance_mask_views=[
                PartialInstanceMaskViewInput(
                    referenceView=create_reference_view(
                        image=image,
                        c_min=0,
                        c_max=1,
                    ),
            ),
            ],
            # dataset=image.dataset,
        )
    
    # Extract and store flow components
    flow_fields = flows[0]  # Shape: [2, H, W] - X and Y flow components
    cellprob = flows[1]     # Shape: [H, W] - Cell probability map
    
    print(f"Flow fields shape: {flow_fields.shape}")
    print(f"Cell probability shape: {cellprob.shape}")
    
    # Create flow field image (combining X and Y flows into RGB visualization)
    if len(flow_fields.shape) == 3:
        if flow_fields.shape[0] == 2:
            # Format: (2, H, W) - need to transpose to (H, W, 2) and add magnitude
            flow_magnitude = np.sqrt(flow_fields[0]**2 + flow_fields[1]**2)
            flow_rgb = np.stack([flow_fields[0], flow_fields[1], flow_magnitude], axis=0)  # Shape: [3, H, W]
            # Transpose to (H, W, 3) for xarray
            flow_rgb = flow_rgb.transpose(1, 2, 0)  # Now (H, W, 3)
            flow_rgb = (flow_rgb - flow_rgb.min()) / (flow_rgb.max() - flow_rgb.min() + 1e-8)  # Normalize to [0,1]
        elif flow_fields.shape[2] == 3:
            # Format: (H, W, 3) - already in correct format, just normalize
            flow_rgb = flow_fields.copy()
            flow_rgb = (flow_rgb - flow_rgb.min()) / (flow_rgb.max() - flow_rgb.min() + 1e-8)  # Normalize to [0,1]
        elif flow_fields.shape[2] == 2:
            # Format: (H, W, 2) - add magnitude as third channel
            flow_magnitude = np.sqrt(flow_fields[:, :, 0]**2 + flow_fields[:, :, 1]**2)
            flow_rgb = np.concatenate([flow_fields, flow_magnitude[:, :, np.newaxis]], axis=2)
            flow_rgb = (flow_rgb - flow_rgb.min()) / (flow_rgb.max() - flow_rgb.min() + 1e-8)  # Normalize to [0,1]
        else:
            # Unexpected 3D format - just use as is
            flow_rgb = flow_fields
            flow_rgb = (flow_rgb - flow_rgb.min()) / (flow_rgb.max() - flow_rgb.min() + 1e-8)  # Normalize to [0,1]
        
        flow_image = from_array_like(
            xr.DataArray(flow_rgb, dims=("x", "y", "c")),
            name=f"Flow field of {image.name}",
            tags=["cellpose", "SAM", "prediction", "flow"],
        )
    elif len(flow_fields.shape) == 2:
        # Single 2D array
        flow_image = from_array_like(
            xr.DataArray(flow_fields, dims=("x", "y")),
            name=f"Flow field of {image.name}",
            tags=["cellpose", "SAM", "prediction", "flow"],
        )
    else:
        # Fallback - squeeze and hope for the best
        flow_image = from_array_like(
            xr.DataArray(flow_fields.squeeze(), dims=("x", "y")),
            name=f"Flow field of {image.name}",
            tags=["cellpose", "SAM", "prediction", "flow"],
        )
    
    # Create cell probability image - handle different shapes
    if len(cellprob.shape) == 2:
        # Expected 2D case
        cellprob_image = from_array_like(
            xr.DataArray(cellprob, dims=("x", "y")),
            name=f"Cell probability of {image.name}",
            tags=["cellpose", "SAM", "prediction", "cellprob"],
        )
    elif len(cellprob.shape) == 3:
        # 3D case - squeeze or take first channel
        cellprob_2d = cellprob.squeeze() if cellprob.shape[0] == 1 or cellprob.shape[2] == 1 else cellprob[0]
        cellprob_image = from_array_like(
            xr.DataArray(cellprob_2d, dims=("x", "y")),
            name=f"Cell probability of {image.name}",
            tags=["cellpose", "SAM", "prediction", "cellprob"],
        )
    else:
        # Fallback - just squeeze all singleton dimensions
        cellprob_image = from_array_like(
            xr.DataArray(cellprob.squeeze(), dims=("x", "y")),
            name=f"Cell probability of {image.name}",
            tags=["cellpose", "SAM", "prediction", "cellprob"],
        )
    
    # print(styles[0].shape())
    # style = from_array_like(
    #         xr.DataArray(styles, dims=("x", "y")),
    #         name=f"Style of {image.name}",
    #         tags=["cellpose", "SAM", "prediction"],
    #         # dataset=image.dataset,
    #     )
    
    ### Create Graph Relation
    
    return mask, flow_image, cellprob_image

@register(collections=["segmentation","training",])
def train_cellpose_SAM(
    model_name: str = "custom_cellpose_model",
    graph_query: GraphQuery,
    train_test_split: float = 0.8,
    n_epochs = 100,
    learning_rate = 1e-5,
    weight_decay = 0.1,
    batch_size = 1,
    test: bool = False,
    ) -> Model:
    """Trains a custom Cellpose model using the data provided by the graph query.

    Args:
        graph_query (GraphQuery): _description_
        model_name (str, optional): _description_. Defaults to "custom_cellpose_model".
        train_test_split (float, optional): _description_. Defaults to 0.8.
        n_epochs (int, optional): _description_. Defaults to 100.
        learning_rate (_type_, optional): _description_. Defaults to 1e-5.
        weight_decay (float, optional): _description_. Defaults to 0.1.
        batch_size (int, optional): _description_. Defaults to 1.
        test (bool, optional): _description_. Defaults to False.

    Returns:
        Model: _description_
    """
    ###
    # ToDo: Ask @jhnnsrs if this is the correct way to create a graph query
    graph_query = create_graph_query(
        "Cellpose",
        "Pairs of image inside the dataset",
        kind=ViewKind.PAIRS,
        query=f"""
        MATCH (n)-[r]->(m)
        WHERE r.__category_id = {IS_MASK_FOR.id}
        RETURN n, r, m
        """,
        relevant_for=[IS_MASK_FOR],
    )
    ###
    
    # Loading the default model which training is added to
    model = models.CellposeModel(gpu=True)

    # Getting Training Data and Labels from Graph Query
    render = graph_query.render
    assert isinstance(render, Pairs), "Graph query render must be of type Pairs"
    
    train_data, train_labels, test_data, test_labels = [], [], [], []
    split_index = int(train_test_split * len(render.pairs))

    for pair in render.pairs[:split_index]:
        image_structure = pair.source
        mask_structure = pair.target
        train_data.append(cast(Image, image_structure.resolve()))
        train_labels.append(cast(Image, mask_structure.resolve()))
        
    for pair in render.pairs[split_index:]:
        image_structure = pair.source
        mask_structure = pair.target
        test_data.append(cast(Image, image_structure.resolve()))
        test_labels.append(cast(Image, mask_structure.resolve()))

    # Train Custom Cellpose Model
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
