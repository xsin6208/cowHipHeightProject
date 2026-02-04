#!/usr/bin/env python3
import os, json, cv2, numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os.path
from ultralytics import YOLO
from PIL import Image, ImageDraw
from typing import Optional, Tuple, List, Union
Point = Tuple[float, float]
import glob

device = torch.device("cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"))
# print("Using device:", device)

# Cow Detection Model Class Provides the pipeline for segmenting the cow and determining the point where hip height should be measured from
# Also includes methods for testing and determining cow posture and image blurriness
class cowDetectionModel():
    # Initialisation Method
    # Input: Takes in the device processor for where the models will operate on
    # Object variables: Each model will have variables which hold the depth and image mask, the original images, the segmentation model, groin model and YoloModel
    # Constant: The image size sets a constant for the size each image will be resized into before being processed by models
    def __init__(self, device):
        self.device = device
        self.IMG_SIZE = 224
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.SEGMENTATION_MODEL_PATH = os.path.join(script_dir, "lraspp_cow_final.pth")
        self.CKPT_PATH    = os.path.join(script_dir, "resnet50_groin_xy.pth")
        self.yoloModel = YOLO("yolo11x-seg.pt") 
        self.tfm_img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.mask = None
        self.depth = None
        self.depthOriginal = None
        self.segModel = self.segmentationModelSetup()
        self.groinXYModel = self.groinModelSetup()
        self.cropValues = dict()
        print("Using device:", device)

    ##################################################################################################### internal methods ###########################################################################################################
    # Resizes an image to first make it square using padding and then resizes it to default 224 to 224
    def resize_with_padding(
        self,
        img: np.ndarray, # image to be resized
        target_size: int = 224, # desired size (default 224)
        pad_value: Tuple[int, int, int] = (255, 255, 255), # Padding value (all white default)
        xy: Optional[Union[Point, List[Point]]] = None, 
        save_dir: str = "pipeline_img", #this folder where images are saved if desired to use repetitively
        save_name: str = "crop_224.png", # name of image
    ):
        # Get the image height and width
        h, w = img.shape[:2]
        # Determine the larger dimension to make the image square
        max_side = max(h, w)
    
        # Convert grayscale images to 3-channel BGR to maintain consistency
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
        # Create a square white canvas of size (max_side x max_side)
        canvas = np.full((max_side, max_side, 3), pad_value, dtype=img.dtype)
        # Calculate top and left offsets so that the image is centered on the canvas
        top = (max_side - h) // 2
        left = (max_side - w) // 2
        # Place the image in the center of the square canvas
        canvas[top:top + h, left:left + w] = img
    
        # Choose interpolation method: downsampling uses INTER_AREA, upsampling uses INTER_LINEAR
        interp = cv2.INTER_AREA if target_size < max_side else cv2.INTER_LINEAR
        # Resize the square image to the final target dimensions (e.g., 224x224)
        resized_img = cv2.resize(canvas, (target_size, target_size), interpolation=interp)
        # Compute the overall scaling factor relative to the original largest dimension
        scale = target_size / max_side
    
        mapped_xy = None
        if xy is not None:
            # Helper function to transform a single (x, y) coordinate based on padding and scaling
            def _map_one(pt: Point) -> Point:
                x, y = pt
                # Apply translation (left/top padding) and scaling to map to resized image coordinates
                x_new = (x + left) * scale
                y_new = (y + top) * scale
                return (int(round(x_new)), int(round(y_new)))
    
            # Handle both a list of points and a single point
            if isinstance(xy, (list, tuple)) and len(xy) > 0 and isinstance(xy[0], (list, tuple)):
                # Map each point individually
                mapped_xy = [_map_one((float(px), float(py))) for (px, py) in xy]
            else:
                # Single coordinate mapping
                px, py = xy
                mapped_xy = _map_one((float(px), float(py)))
    
        # Return resized image, scaling factor, padding offsets, original size, and optionally mapped coordinates
        return [resized_img, scale, (top, left), (h, w), mapped_xy]

    # Resizes a mask to first make it square using padding and then resizes it to default 224 to 224
    def resize_mask_with_padding(
        self,
        mask: np.ndarray,
        target_size: int = 224,
        pad_value: int = 0,
        xy: Optional[Union[Point, List[Point]]] = None,
    ):
        h, w = mask.shape[:2]
        max_side = max(h, w)

        # Ensure binary 0/1 mask
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        # Create square canvas with zero padding
        canvas = np.full((max_side, max_side), pad_value, dtype=mask.dtype)

        # Compute top-left offsets to center the mask
        top = (max_side - h) // 2
        left = (max_side - w) // 2
        canvas[top:top + h, left:left + w] = mask

        # Resize with nearest neighbor to keep binary values
        interp = cv2.INTER_AREA if target_size < max_side else cv2.INTER_LINEAR
        resized_mask = cv2.resize(canvas, (target_size, target_size), interpolation=interp)
        scale = target_size / max_side

        # Map coordinates (if any)
        mapped_xy = None
        if xy is not None:
            def _map_one(pt: Point) -> Point:
                x, y = pt
                x_new = (x + left) * scale
                y_new = (y + top) * scale
                return (int(round(x_new)), int(round(y_new)))

            if isinstance(xy, (list, tuple)) and len(xy) > 0 and isinstance(xy[0], (list, tuple)):
                mapped_xy = [_map_one((float(px), float(py))) for (px, py) in xy]
            else:
                px, py = xy
                mapped_xy = _map_one((float(px), float(py)))

        return [resized_mask, scale, (top, left), (h, w), mapped_xy]

    # Transforms a polygon outline shape into a filled in mask
    def polygons_to_mask(self, polygons, h=224, w=224):
        mask = np.zeros((h, w), dtype=np.uint8)
        if polygons:
            cv2.fillPoly(mask, [np.array(p, dtype=np.int32) for p in polygons], 255)
        return mask

    # Prototype function not used in final integration
    # Recursive function which finds the point where the backlegs split from the body by using Y cuts and checking if the leftmost point is past a threshold
    def binarySearchLegCut(self, backLegs, top, bottom):
        yCut = (top + bottom) // 2
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(backLegs[yCut:bottom,:], connectivity=8)
        leftMost = np.nonzero(backLegs[yCut,:])[0][0]
        rightMost= np.nonzero(backLegs[yCut,:])[0][-1]
        threshold = (np.floor(0.3*(rightMost))).astype(np.uint8)
        # Checks if the back of the cow at the Y cut has its leftmost point past the threshold which would indicate legs not body
        # Uses recursion to find the last point where this occurs - therefore where the legs start in the y axis
        if abs(top - bottom) < 2:
            return bottom
        elif np.all(leftMost > threshold):
            return self.binarySearchLegCut(backLegs, top, (bottom - top)//2 + top)
        else:
            return self.binarySearchLegCut(backLegs, (bottom - top)//2 + top, bottom)


    # Function determines a score for the blurriness of the image using variance of Laplacians
    # Not required in final integration due to high standard of ToF camera 
    def blurLaplacianVar(self, img, mask=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        if mask is not None:
            lap = lap[mask > 0]
        score = lap.var()
        return float(score)

    # Prototype function not used in final integration
    # Removes the head of the cow from the rest of the segmentation
    def beheadCow(self, mask, top, bottom):
        leftCut, rightCut = self.findBody(mask, bottom-20, bottom)
        return mask[:, leftCut:rightCut+1]

    # Prototype function not used in final integration
    # Finds where the connected components when moving up the cow changes from numerous (legs) to 1 (whole body) and uses the x range of the legs to obtain this part of image
    # Currently doesn't work if head is bent down in line with legs (assumes its a leg)
    def findBody(self, mask, top, bottom):
        numLabels, labels = cv2.connectedComponents(mask[top:bottom+1,:], connectivity=8)
        if numLabels == 2:
            ys, xs = np.nonzero(mask[top+5:,:])
            leftmost  = xs.min()
            rightmost = xs.max()
            return leftmost, rightmost
        else:
            return self.findBody(mask, top-5, bottom)

    # Prototype function not used in final integration
    # Used with a cropping of just the backlegs of a cow, determines how straight the legs are using a best fit 
    def determineLegStraightness(self, backLegsOnly, yCut, bottom):
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(backLegsOnly[yCut:bottom+1,:], connectivity=8)
        leg_mask = (labels == 1).astype(np.uint8)

        ys, xs = np.where(leg_mask > 0)
        pts = np.column_stack((xs, ys)).astype(np.float32)

        mu = pts.mean(axis=0)
        X = pts - mu
        # SVD for principal axis
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        u = Vt[0]                    # first principal direction (unit vector)

        # orthogonal distances to the best-fit line
        proj = (X @ u[:,None]) * u   # projected component
        ortho = X - proj
        d = np.linalg.norm(ortho, axis=1)

        # normalize by leg length along the axis
        leg_len = proj.max() - proj.min()
        straightness = 1.0 - (d.std() / max(leg_len, 1e-6))  # in [~0,1], higher = straighter
        angle_deg = np.degrees(np.arctan2(u[1], u[0]))
        print(straightness)

    # Finds how many connected components there are in the image and if multiple chooses the largest as the primary cow which is returned as mask
    # If no cow (only background) returns -1, None to represent this
    # Also returns -1 None if cow is at edge of image (therefore cut out by frame)
    def getPrimaryCow(self, mask):
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # if cow in image
        if numLabels >= 2:
            # Find biggest cow
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_idx = np.argmax(areas)
            x, y, w_box, h_box, area = stats[max_idx + 1]
            # check if cow on edge of frame
            touches_edge = (
                x == 0 or
                x + w_box >= mask.shape[1]
            )
            # if cow is too small don't include or if it is on edge of frame
            if areas[max_idx] < 100 or touches_edge:
                return -1, None
            mask = np.where(labels == max_idx + 1, 255, 0).astype(np.uint8)
        else:
            return -1, None
        # returns mask and non error 0 
        return 0, mask

    # Sets up the segmentation model by reading it from the file in which its saved
    def segmentationModelSetup(self):
        #load model
        model = models.segmentation.deeplabv3_resnet50(weights=None)
        model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)

        state_dict = torch.load(self.SEGMENTATION_MODEL_PATH, map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier")}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Seg] Loaded with {len(missing)} missing and {len(unexpected)} unexpected keys "
              f"(aux head ignored).")

        model.to(self.device).eval()
        return model

    # Crops the cow segmentation mask and overlay image to only include the segmentation and 10 padding pixels on all sides
    def cropSegment(self, mask, overlay):
        H,W = mask.shape[0:2]
        ys, xs = np.where(mask > 0)
        y1, y2 = max(0, ys.min()-10), min(H, ys.max()+10)
        x1, x2 = max(0, xs.min()-10), min(W, xs.max()+10)

        crop_rgb  = overlay[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        self.cropValues["x1"] = x1
        self.cropValues["x2"] = x2
        self.cropValues["y1"] = y1
        self.cropValues["y2"] = y2

        return crop_mask, crop_rgb

    # Function which runs the model on the cow image and gets a cow segmentation mask and cropped image
    # Has input of image and also yolo which can be set to true or false to use either Yolo model or self created segmentation model
    def segmentCow(self, img, yolo = False):
        # preprocess image (resizing to 224 by 224)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, *_ = self.resize_with_padding(img_rgb)

        # plt.imshow(img)
        # plt.show()
        # Runs image in model to get a binary mask
        if yolo:
            pred_mask_bin, *_ = self.resize_mask_with_padding(self.getTrueVals(img))

        else:
            with torch.no_grad():
                input_tensor = self.tfm_img(img).unsqueeze(0).to(device)
                pred_logits = self.segModel(input_tensor)["out"]
                pred_mask = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()


            pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255

        # saves mask to object variable
        self.mask = pred_mask_bin.copy()

        # check for cow in image and also remove any extra cows (smaller cows are removed)
        cow, pred_mask_bin = self.getPrimaryCow(pred_mask_bin)

        # No cow found in image or cow on edge of image
        if cow == -1:
            return None

        # Perform morphological clean up of segmentation
        kernel = np.ones((3,3), np.uint8)
        pred_mask_bin = cv2.morphologyEx(pred_mask_bin, cv2.MORPH_OPEN,  kernel, iterations=1)
        pred_mask_bin = cv2.morphologyEx(pred_mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Ensures image is still binary after morphology
        maskBinary = (pred_mask_bin > 0).astype(np.uint8)

        # Create an image with cow segmentation and background set to white
        overlay = img.copy()
        overlay[maskBinary == 0] = [255, 255, 255]

        # crop mask (binary) and segmentation image (coloured)
        crop_mask, crop_rgb = self.cropSegment(maskBinary, overlay)

        # plt.imshow(crop_rgb)
        # plt.show()
        # return cropped segmentation image (coloured)
        return crop_rgb

    # Performs segmentation on image using Yolo
    def getTrueVals(self, img):
        segmentImg = img.copy()
        #model = YOLO("yolo11x-seg.pt")
        results = self.yoloModel(img)

        # Yolo gets outline of image which is collected
        polygons = []
        for result in results:
            if result.masks is None:
                continue
            xyn = result.masks.xyn
            cls = result.boxes.cls
            height, width = segmentImg.shape[:2]
    

            for mask_coords, cls_index in zip(xyn, cls):
                if cls_index!=19:
                    continue
                mask_coords_pixel = mask_coords * np.array([width, height])
                mask_coords_pixel = mask_coords_pixel.astype(np.int32)
                mask_coords_pixel = mask_coords_pixel.reshape(-1, 1, 2)
                cv2.polylines(segmentImg, [mask_coords_pixel], isClosed=True, color=(0, 255, 0), thickness=2)
                polygons.append(mask_coords_pixel)

        # outline turned into mask
        mask = self.polygons_to_mask(polygons, h=segmentImg.shape[0], w=segmentImg.shape[1])

        # #plot result
        # plt.figure(figsize=(8,6))
        # plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()
        
        # mask returned
        return mask

    # Performance function which calculates IOU, Recall, Precision, F1 and accuracy for an input estimation and truth values
    def calculatePerformance(self, trueMask, predMask):
        truePositives = np.sum( (trueMask > 0) & (predMask > 0) )

        falsePositives = np.sum( (trueMask == 0) & (predMask > 0) )

        falseNegatives = np.sum( (trueMask > 0) & (predMask == 0) )

        trueNegatives = np.sum( (trueMask == 0) & (predMask == 0) )

        iou = truePositives/(falseNegatives + falsePositives + truePositives + 1e-8)
        recall = truePositives / (truePositives + falseNegatives + 1e-8)
        precision = truePositives / (truePositives + falsePositives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (truePositives + trueNegatives) / (
            truePositives + falseNegatives + falsePositives + trueNegatives + 1e-8
        )
        #print("IoU: {:.3f}, recall: {:.3f}, precision: {:.3f}, f1: {:.3f}".format(iou, recall, precision, f1))
        return iou, recall, precision, f1

    # Tests the performance of the segmentation model against Yolo getting IOU, recall, precision and F1
    def segTest(self, img):
        crop = self.segmentCow(img)
        trueMask, *_ = self.resize_mask_with_padding(self.getTrueVals(img))
        iou, recall, precision, f1 = self.calculatePerformance(trueMask, self.mask)
        print("IoU: {:.3f}, recall: {:.3f}, precision: {:.3f}, f1: {:.3f}".format(iou, recall, precision, f1))
        return iou, recall, precision, f1
        
    # Does a segmentation test for a whole folder, and produces average results for the whole folder
    def segTestFolder(self, img_dir):
        img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) \
           + glob.glob(os.path.join(img_dir, "*.png")) \
           + glob.glob(os.path.join(img_dir, "*.jpeg"))
        iou, recall, precision, f1 = 0,0,0,0
        for path in img_paths:
            img = cv2.imread(path)
            iouRet, recallRet, precisionRet, f1Ret = self.segTest(img)
            iou += iouRet
            recall += recallRet
            precision += precisionRet
            f1 += f1Ret
        iou = iou/len(img_paths)
        recall = recall/len(img_paths)
        precision = precision/len(img_paths)
        f1 = f1/len(img_paths)
        print("IoU: {:.3f}, recall: {:.3f}, precision: {:.3f}, f1: {:.3f}".format(iou, recall, precision, f1))

    # Class for the neural network of the groin locater model (structure of network no weights)
    class GroinRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = models.resnet50(weights=None)
            in_feats = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.head = nn.Sequential(
                nn.Linear(in_feats, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, 2),
            )
        def forward(self, x):
            return self.head(self.backbone(x))

    # Sets up the groin model using the file with saved weights and class above
    def groinModelSetup(self):
        model = self.GroinRegressor().to(device)
        state = torch.load(self.CKPT_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    # Runs the groin model on an image which should be segmented using the above segmentation functions
    def findGroin(self, crop):
        # Preprocesses image
        crop, *_ = self.resize_with_padding(crop)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # ---- transforms (match training) ----
        tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # harmless; image is already 224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        
        # ---- inference ----
        H, W = crop.shape[:2]
        assert (H, W) == (224, 224), f"Expected 224x224 after padding, got {H}x{W}"
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            inp = tfm(crop).unsqueeze(0).to(device)
            xy_norm = self.groinXYModel(inp)[0].cpu().numpy()
        xy_norm = np.clip(xy_norm, 0.0, 1.0)
        x_px, y_px = int(xy_norm[0] * W), int(xy_norm[1] * H)

        # ---- visualize & confirm sizes ----
        # plt.figure(figsize=(6,6))
        # plt.imshow(crop)
        # plt.scatter([x_px], [y_px], c='red', s=50)
        # plt.title(f"Groin on 224×224 (pred: {x_px}, {y_px})\nImage size: {W}×{H}")
        # plt.axis("off")
        # plt.show()

        # returns the result of located pixel and image
        return x_px, y_px, crop

    # Tests the groin model vs an input true value - finds the RMS error
    def groinTest(self, img, trueX, trueY):
        imgResized, *_ = self.resize_with_padding(img)
        mask = self.getTrueVals(imgResized.copy())
        cow, mask = self.getPrimaryCow(mask)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


        overlay = imgResized.copy()
        overlay[mask == 0] = [255, 255, 255]
        # plt.imshow(overlay)
        # plt.show()
        crop_mask, crop_rgb = self.cropSegment(mask, overlay)
        x_px, y_px, finalImg = self.findGroin(crop_rgb)
        x_px, y_px = self.translatePointToOriginal(x_px, y_px, img, finalImg)
        # plt.figure(figsize=(6,6))
        # plt.imshow(img)
        # plt.scatter([x_px], [y_px], c='red', s=50)
        # plt.axis("off")
        # plt.show()
        # print(x_px, y_px)
        print(np.sqrt((trueX-x_px)**2+(trueY-y_px)**2))
        return np.sqrt((trueX-x_px)**2+(trueY-y_px)**2)



    # Helpers to parse/look up
    def parse_xy_str(self, s: str):
        s = str(s).strip().replace(" ", "").strip("()")
        x_str, y_str = s.split(",")
        return int(x_str), int(y_str)

    # Used for datagrame to get x and y values by name of image
    def get_xy_by_name(self, df: pd.DataFrame, image_name: str):
        row = df.loc[df["image name"] == image_name, "224 groin xy"]
        if len(row) == 0:
            return None
        return self.parse_xy_str(row.values[0])

    # tests whole folder of images against a CSV file of true values of groin locations
    def groinTestFolder(self, imgFolder, CSVFile):
        df = pd.read_csv(CSVFile, header=None)
        df.columns = ["image name", "dimension", "groin xy", "224 groin xy"]
        df.head()
        results = []
        for (_, row) in df.iterrows():
            path = os.path.join(imgFolder, row["image name"])
            img = cv2.imread(path)
            trueX, trueY = self.get_xy_by_name(df, row["image name"])
            results.append(self.groinTest(img, trueX, trueY))
        print(np.mean(results))

    # Used to resize a monochromatic image to 224 by 224 with padding to make square
    def resizeMono(self, img):
        target_size = 224
        pad_value = 0
        
        h, w = img.shape[:2]
        max_side = max(h, w)

        canvas = np.full((max_side, max_side), pad_value, dtype=img.dtype)
        top = (max_side - h) // 2
        left = (max_side - w) // 2
        canvas[top:top + h, left:left + w] = img

        interp = cv2.INTER_AREA if target_size < max_side else cv2.INTER_LINEAR
        resized_img = cv2.resize(canvas, (target_size, target_size), interpolation=interp)
        
        return resized_img

    # Augments the depth image the same way the groin point image will be augmented after run through models
    def findSimilarDepth(self, img):
        img = self.resizeMono(img)
        img = img[self.cropValues["y1"]:self.cropValues["y2"], self.cropValues["x1"]:self.cropValues["x2"]]
        img = self.resizeMono(img)
        return img

    # Using a point on the final image where the groin was found, finds the corresponding equivalent point on the original image input to pipeline
    def translatePointToOriginal(self, x_px, y_px, ogImg, finalImg):
        # get x and y dimensions before resizing and scaling
        xDim = self.cropValues["x2"] - self.cropValues["x1"]
        yDim = self.cropValues["y2"] - self.cropValues["y1"]

        # Find equivalent point before rescaling to 224 by 224 the second time (before groin model)
        scale = max(xDim, yDim)
        scaleX = np.round(x_px/224 * scale)
        scaleY = np.round(y_px/224 * scale)

        # finds point before padding was added (either to width or height to make square) second time (before groin model)
        if xDim > yDim:
            newX = scaleX
            newY = np.round(scaleY - (xDim-yDim)/2)
        else:
            newY = scaleY
            newX = np.round(scaleX - (yDim-xDim)/2)

        # finds point before cropping was complete on image (after segmentation)
        x_px2 = newX + self.cropValues["x1"]
        y_px2 = newY + self.cropValues["y1"]

        # finds point for image before resized to 224 by 224 first time (before segmentation)
        largerDim = np.argmax(ogImg.shape[0:2])
        scale = ogImg.shape[largerDim]
        scaleX = np.round(x_px2/224 * scale)
        scaleY = np.round(y_px2/224 * scale)

        # Finds point for image before padding added first time (before segmentation
        # Y Dimension larger, x dimension would be padded so has to remove padding pixels
        if largerDim == 0:
            newX = np.round(scaleX - (ogImg.shape[0]-ogImg.shape[1])/2)
            newY = scaleY
        else:
            # X Dimension larger, y dimension would be padded so remove padding there
            newX = scaleX
            newY = np.round(scaleY - (ogImg.shape[1]-ogImg.shape[0])/2)

        # returns corresponding original point
        return newX, newY

    # Projects the point of groin up to the back of the cow, uses a vector to deterine the angle up from the ground plane
    def getHipHeightPoint(self, x_px, y_px, ogImg, finalImg, normVector):
        # If normal vector to ground plane is pointing downwards, flip
        if normVector[1] < 0:
            normVector[1] = -1 * normVector[1]
            normVector[0] = -1 * normVector[0]

        # Find angle of normal plane in x y direction
        angle = np.atan(normVector[0]/normVector[1])*180/np.pi
        # plt.figure(figsize=(6,6))
        # plt.imshow(finalImg)
        # plt.scatter([x_px], [y_px], c='red', s=50)
        # plt.axis("off")
        # plt.show()
        
        (h, w) = finalImg.shape[:2]
        center = (w/2, h/2)
        # rotate image to deal with angle so that up is up in real world frame
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rot = cv2.warpAffine(finalImg, M, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        # Find x point of x_px for new rotated image
        xRot = int(round(M[0,0]*x_px + M[0,1]*y_px + M[0,2]))

        # Find the top point of segmentation directly up from xRot
        yTopEdge = np.where(rot[:, xRot] != 255)[0][0]
        

        # plt.figure(figsize=(6,6))
        # plt.imshow(rot)
        # plt.scatter([xRot], [yTopEdge], c='red', s=50)
        # plt.axis("off")
        # plt.show()

        # Transform points back to original unrotated image for height point
        M_inv = cv2.invertAffineTransform(M)

        xHip = int(round(M_inv[0,0]*xRot + M_inv[0,1]*yTopEdge + M_inv[0,2]))
        yHip = int(round(M_inv[1,0]*xRot + M_inv[1,1]*yTopEdge + M_inv[1,2]))

        # plt.figure(figsize=(6,6))
        # plt.imshow(finalImg)
        # plt.scatter([xHip], [yHip], c='red', s=5)
        # plt.axis("off")
        # plt.show()

        # If depth data is available, move hip height point downwards if gone past cow determined by depth being outside of outlier range or 0
        if self.depth is not None:
            yHip = self.outlierDetection(xHip, yHip, finalImg)

        # plt.figure(figsize=(6,6))
        # plt.imshow(finalImg)
        # plt.scatter([xHip], [yHip], c='red', s=5)
        # plt.axis("off")
        # plt.show()

        # Transform final image hip height point to original input image hip height point
        newX, newY = self.translatePointToOriginal(xHip, yHip, ogImg, finalImg)

        # Ensure depth value isn't zero and if it is move down till not (likely due to being too far above cow) 
        if self.depth is not None:
            newY = self.getValidDepth(int(newX), int(newY), ogImg)
        #---- visualize & confirm sizes ----
        # plt.figure(figsize=(6,6))
        # plt.imshow(finalImg)
        # plt.scatter([xHip], [yHip], c='red', s=50)
        # plt.axis("off")
        # plt.show()
        # plt.figure(figsize=(6,6))
        # plt.imshow(self.depth)
        # plt.scatter([xHip], [yHip], c='red', s=50)
        # plt.axis("off")
        # plt.show()
        # plt.figure(figsize=(6,6))
        # plt.imshow(ogImg)
        # plt.scatter([newX], [newY], c='red', s=50)
        # plt.axis("off")
        # plt.show()
        return newX, newY

    # Moves point down till depth point is a proper value so height for this point can be found
    def getValidDepth(self, xHip, yHip, img):
        while self.depthOriginal[yHip, xHip] <= 1:
            yHip += 1
        return yHip

    # Finds the statistical mean and std dev of cow seg depth points and ensures that the final point is not an outlier (where the segmentation is slightly greater than actual cow and has outliers on edge)
    # Does this by moving point down till not an outlier (on that edge just outside actual cow depth)
    def outlierDetection(self, xHip, yHip, finalImg):
        mask = np.all(finalImg == [255, 255, 255], axis=-1)
        self.depth[mask] = 0
        depthValues = self.depth[self.depth != 0]
        std = np.std(depthValues)
        mean = np.mean(depthValues)
        while abs(self.depth[yHip, xHip] - mean) > 2*std or self.depth[yHip, xHip] <= 1:
            yHip += 1
        return yHip
        
    ####################################################################################### EXTERNAL METHOD: IMPLEMENTATION ############################################################################################# 
    
    # Pipeline for image segmentation and get hip height point on original image, returns -1, -1 if no cow or cow on edge of frame or the hip height point determined when there is cow
    # Inputs: 
    #    img: the image frame from video which is desired to have cow segmented and hip height point extracted
    #    depth: depth image of same frame - must be already aligned with image and same resolution (default depth not required but won't work nearly as well)
    #    normVector: Normal vector of ground plane of scene (z coordinate is in direction of camera sight), default is a vector straight upwards
    #    useYolo: Binary input to choose whether to use yolo model or own created segmentation model stored in file (default use own model not yolo)
    # Return:
    #     -1, -1 - No cow found or cow on edge of image
    #     int, int = cow hip height point to calculate distance from ground to
    def hipHeightPipeline(self, img, depth = None, normVector = (0,1,0), useYolo = False):
        self.depthOriginal = depth.copy()
        self.depth = depth
        crop = self.segmentCow(img, useYolo)
        if crop is None:
            return -1, -1
        x_px, y_px, finalImg = self.findGroin(crop)
        if self.depth is not None:
            self.depth = self.findSimilarDepth(self.depth)
        return self.getHipHeightPoint(x_px, y_px, img, finalImg, normVector)

