# A synthesis on instance segmentation
## Single Stage
###  Embed Mask 
<img src="resources\EmbedMask.png" style="zoom:50%;" />

<p style="font-size:60%;text-align:center;">▲Embed Mask architecture: FCOS + Pixel Embedding.</p>

#### Motivation
- Tap the high recognition ability of top-down detectors and high resolution of bottom-up segmentors.
#### Methods: 
FCOS  with a **Pixel Embedding** Head to obtain fine instance masks.

- Obtain instance masks by calculating distances between **pixel embedding $p_i$** and  **proposal embedding $Q_k$**, which originates from different layers.  (shown in the figure above)
$$
\operatorname{Mask}_{k}\left(x_{i}\right)=\left\{\begin{array}{ll}
  1, & \left\|p_{i}-Q_{k}\right\| \leq \delta \\
  0, & \left\|p_{i}-Q_{k}\right\|>\delta
  \end{array}\right.
$$


- Design a learnable margin to choose $\delta$

  $$
  \phi\left(x_{i}, S_{k}\right)=\phi\left(p_{i}, Q_{k}, \Sigma_{k}\right)=\exp \left(-\frac{\left\|p_{i}-Q_{k}\right\|^{2}}{2 \Sigma_{k}^{2}}\right),
  $$

  where the $\Sigma_k$ plays a role of margin for instance $S_k$. **Note that the bbox regression head actually predicts $\frac{1}{2\sigma^2}$ instead of $\sigma!$**
  
- Introduce smooth loss to diminish the gap of proposal embeddings during train/inference. (yet lacks ablation study support)

#### Results

<img src="resources\EmbedMask_Exp.png" style="zoom:50%;" />

<p style="font-size:60%;text-align:left;">It claims to be the first single-stage instance segmentation network to outperform Mask-RCNN under the same training settings.</p>


###  Tensor Mask

#### Motivation

Inspired by DeepMask, it predicts a $M \times N$ mask densely, resulting into a 4D prediction of size $M \times N \times H \times W$.

<img src="resources\TensorMask.png" 
align="left" hspace="50px" style="zoom:50%;"/><img src="resources\TensorMask2.png" 
align="left" style="zoom:50%;"/>











<p style="font-size:60%;text-align:center;">L: Aligned representation is organized. R:FPN multi-level predictions are upsampled and composed into tensor bipyramid using swap_align2nat.</p
#### Method

- propose an "aligned" version of the 4D dense prediction.

- propose `up_align2nat` and `swap_align2nat` operations that are specially designed for the "aligned" representation. These operations are used to up-sample masks and compose multi-scale tensor pyramids.

#### Results

<img src="resources\TensorMask_Exp.png" style="zoom:67%;" />

mAP is comparable with Mask-RCNN~(slightly inferior). However, due to its $M \times N$ prediction and expensive upsampling operation, the single stage solution runs slower~(3FS) than the two-stage Mask-RCNN~(11FPS).

### ESE-Seg

#### Motivation

Through the parameterization of the shape of instances, the network can output instances masks by predicting the encoded parameters, which cost much less time than dense masks.

#### Method

1. It first locates the center of an instance $(x, y)$, and generate N rays from its center, obtaining N rays of length $\{r_1, \cdots, r_i, \cdots, r_N\}$ at $\theta_i$, where $\theta_i=i \frac{2\pi}{N}$. Now, the presentation of an instances becomes $\{x, y, r_1, ..., r_n\}$
2. To facilitate the learning of $r_i$, the author models $r_i = f(\theta_i)$ as a Chebyshev polynomials:  $\boldsymbol{f}(\theta) \sim \sum_{i=0}^{\infty} c_{i} T_{i}(\theta)$, where

$$
\begin{aligned}
T_{0}(x)&=1 \\
T_{1}(x)&=x \\
T_{n+1}(x)&=2 x T_{n}(x)-T_{n-1}(x).
\end{aligned}
$$

#### Results

<img src="resources\ESE_Exp.png" 
align="left" style="zoom:80%;"/><img src="resources\ESE_Vis.png" 
align="left" style="zoom: 55%;"/>









It runs at 38 FPS, with a mAP of 21.6. Unsurprisingly, the visualization shows that mask predictions tend to be over-smoothed.

### FourierNet

#### Motivation

The same as ESE-Seg except that it parameterizes $\{r_1, ..., r_n\}$ through discrete fourier transform.

#### Results

<img src="resources\FourierNet.png" style="zoom:50%;" />

Both mAP and FPS are inferior to PolarMask.

###  PolarMask

#### Motivation

The same as ESE-Seg except that it directly regresses $\{r_1, ..., r_n\}$.

#### Method

FCOS + Polar centerness & ray length Prediction.

To better learn the $r_i$, it proposes two key parts:

1. Polar Centerness: 
   
   similar to the use of centerness in FCOS, but designed for a polygon:
   $$
\begin{equation}\text { Polar Centerness }=\sqrt{\frac{\min \left(\left\{d_{1}, d_{2}, \ldots, d_{n}\right\}\right)}{\max \left(\left\{d_{1}, d_{2}, \ldots, d_{n}\right\}\right)}}\end{equation}
   $$
   
2. Polar IoU Loss: 

   instead of regressing $r_i$ independently, this work uses Polar IoU Loss to model the correlation of $\{r_1, ..., r_n\}$
<img src="resources\PolarMask.png" 
   align="left" style="zoom: 75%;"/>
   $$
   \begin{equation}\mathrm{IoU}=\frac{\int_{0}^{2 \pi} \frac{1}{2} \min \left(d, d^{*}\right)^{2} d \theta}{\int_{0}^{2 \pi} \frac{1}{2} \max \left(d, d^{*}\right)^{2} d \theta}\end{equation}
   $$
   $$
   \begin{equation}\mathrm{IoU}=\lim _{N \rightarrow \infty} \frac{\sum_{i=1}^{N} \frac{1}{2} d_{\min }^{2} \Delta \theta_{i}}{\sum_{i=1}^{N} \frac{1}{2} d_{\max }^{2} \Delta \theta_{i}}\end{equation}
   $$
   $$
   \begin{equation}\text { Polar IoU }=\frac{\sum_{i=1}^{n} d_{\min }}{\sum_{i=1}^{n} d_{\max }}\end{equation}
   $$

   

#### Results

<img src="resources\PolarMask_Exp.png" 
align="left" style="zoom:72%;"/><img src="resources\PolarMask_Vis.png" 
align="left" style="zoom: 28%;"/>







It runs at 17.2FPS with a mAP of 29.1. Notice that the mask predictions are much crisper than ESE-Seg.

### YOLOACT

<img src="resources\YOLOACT.png" style="zoom:67%;" />

#### Methods

It predicts k~(constant) prototypes~(mask templates) and coefficients of these prototypes, the final instance mask is weighted sum of these protos.

- the prototypes $P\in R^{H\times W\times K}$ are obtained by feeding P3 into a FCN. There is no direct supervision on the prototypes. Authors consider it vial to keep the output of FCN **unbounded**.
- the coefficients $C \in R^k$ are obtained from the classic bbox regression head. Activation function is `tanh` to enable negative values.
- the  final mask is $ P = \sigma(P C^T)$. The author then crops this mask using the predicted bbox. (during training, only losses within the GT bbox are counted.)

#### Results

![](resources\YOLOACT_Exp.png)

#### Insights

1. The authors tried to use a classic FC layer to directly predict the instance mask~(flatten). It only obtains 20.7mAP, indicating that it is insufficient using a single location's feature to predict the whole instance map. **Contradictory to TensorMask**
2. This pipeline shows that FCN itself **can be translation variant**: if  FCN was translation invariant, YOLOACT should fail when there are two identical instances in the image.

### YOLOACT++

#### Methods

- It designs a mask re-scoring head, feed the $H\times W \times K$ masks into a 6-layer GAP to regress its IoU with GT mask. This IoU is used to reweight the masks.
- Apply deformable convolutions 
- Add more anchor settings

#### Results

![](resources\YOLOACT_PLUS_Exp.png)

###  SOLO

#### Methods

![](resources\SOLO.png)

- SOLO quantizes locations by dividing images into $S\times S$ grids. Each grid is responsible to predict only one instance. To do so, it designs two branches: category branch and mask branch. 
- The mask branch predicts a tensor $M$ of shape $H \times W \times S^2$.  For each pixel located at $(x,y)$, the corresponding vector $M[i, j] \in R^{S^2}$represents values at $(x, y)$ originating from $S \times S$ grids. This formation is similar to the idea of `aligned representation` in TensorMask. 
- To enable each position to  speculate the mask prediction originating from grids all over the image, the mask head mask be deep enough, and also positionally sensitive. This the author applies mask head of depth = 7 and `CoordConv`.

#### Results

#### <img src="resources\SOLO_Exp.png" style="zoom:50%;" />

It achieves a similar performance as Mask-RCNN by adopting a longer 6x training schedule. SOLO Res50-FPN runs at 12.1FPS.

## Two-Stage

### DeepSnake

#### Motivation

Its motivation is the same as ESE-Seg except that it implements the task of contour regression using a learning-based snake algorithm, resulting in a two-stage solution.

#### Methods

It proposes a DeepSnake modules which does:

1. regress bbox vertexes to extreme points, thus turn the initial box into a octagon.
2. interpolate values at vertexes of the octagon and go through $8$ `circonv` layers to predict offsets of vertexes. This step is applied iteratively~(3 times).

#### Results

No results given on COCO. Experiments on other datasets show that it is better than Mask-RCNN and achieves similar performance as PANet with a 5x running speed.

<img src="resources\DeepSnake.png" style="zoom:50%;" />

### PointRend

<img src="resources\PointRend.png" style="zoom:50%;" />

<p style="font-size:60%;text-align:left;"> To refine the coarse mask, PointRend selects a set of points (red dots) and makes prediction for each point independently with a small MLP. The MLP uses interpolated features computed at these points (dashed red arrows) from (1) a fine-grained feature map of the backbone CNN and (2) from the coarse prediction mask. The coarse mask features enable the MLP to make different predictions at a single point that is contained by two or more boxes. The proposed subdivision mask rendering algorithm (see Fig. 4 and §3.1) applies this process iteratively to refine uncertain regions of the predicted mask.
3.</p>
#### Motivation

Inspired by Adaptive Subdivision in CG, PointRend refines predictions at uncertain locations and obtains fine mask predictions by applying refinement from iteratively.

#### Method

It proposes a PointRend Module composing three main parts:

1. A point selection strategy: locate suspicious points
2. A point-wise feature extractor: extract points features using interpolation.
3. A point head~(MLP) to refine predictions for suspicious points.

During inference, the PointRend upsamples predictions 2x each time by:

1. upsample predictions by vanilla bilinear interpolation
2. select N uncertain points~(those with scores close to 0.5 for binary mask).
3. compute their values and re-predict them using the MLP.

During training, to avoid the iterative upsampling process, sampling points~(float coordinates) are chosen following a proposed non-iterative random sampling strategy.

#### Results

<img src="resources\PointRend_Exp.png" style="zoom:50%;" />

The results shows that PointRend generates more detailed masks. It runs at 13FPS, close to Mask-RCNN due to use its lighter mask head.

### BlendMask

#### Method

It is actually an incremental improvement of YOLOACT. Recall that YOLOACT crop s instance masks with their boxes before composing them. BlendMask replaces this by applying attention. However, predicting an attention map $H\times W\times K \times H \times W$ densely can be intractable. BlendMask thus extends YOLOACT into a two-stage solution and predicts attention map for the $7 \times 7$ cropped region.

#### Results

<img src="resources\BlendMask_Exp.png" style="zoom:50%;" />

It doesn't show large advantage over Mask-RCNN under the same training setting.

 

