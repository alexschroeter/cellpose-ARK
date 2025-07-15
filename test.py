import numpy as np
import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline - commented out as this is a Jupyter magic command
mpl.rcParams['figure.dpi'] = 200

# Add cellpose directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "cellpose"))

from cellpose import utils, io


def test_run_cellpose3_notebook():
    """
    Test the functionality of the cellpose3 notebook example
    https://github.com/MouseLand/cellpose/blob/main/notebooks/run_cellpose3.ipynb

    This function downloads a set of noisy images, runs the Cellpose denoising model,
    and visualizes the results including the denoised images and segmentation outlines.
    """

    # download noisy images from website
    url = "http://www.cellpose.org/static/data/test_poisson.npz"
    filename = "test_poisson.npz"
    utils.download_url_to_file(url, filename)
    dat = np.load(filename, allow_pickle=True)["arr_0"].item()
    
    # Get the test images
    imgs = dat["test_noisy"]

    imgs = dat["test_noisy"]
    # plt.figure(figsize=(8,3))
    # for i, iex in enumerate([2, 18, 20]):
    #     img = imgs[iex].squeeze()
    #     plt.subplot(1,3,1+i)
    #     plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    ##########################################################

    # RUN CELLPOSE3

    from cellpose import denoise, io

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

    ##########################################################

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

    ##########################################################

    # ToDo Compare Results

    # assert "Images are the same"

def run_cellpose_tests():
    """
    Run all pytest tests in the cellpose/tests directory.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    import subprocess
    import sys
    import os
    from pathlib import Path
    
    # Get the cellpose directory path
    cellpose_dir = Path(__file__).parent / "cellpose"
    tests_dir = cellpose_dir / "tests"
    
    if not tests_dir.exists():
        print(f"Error: Tests directory not found at {tests_dir}")
        return False
    
    # Change to the cellpose directory to run tests
    original_cwd = os.getcwd()
    
    try:
        os.chdir(cellpose_dir)
        
        # Run pytest on the tests directory
        cmd = [
            sys.executable, "-m", "pytest", 
            str(tests_dir),
            "-v",  # verbose output
            "--tb=short",  # shorter traceback format
            "--disable-warnings",  # disable warnings for cleaner output
        ]
        
        print(f"Running tests in {tests_dir}...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Tests timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    # Run the cellpose tests
    success = run_cellpose_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

