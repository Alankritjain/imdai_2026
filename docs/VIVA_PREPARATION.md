# Viva preparation: 25 questions and answers

1. **What is behavioral cloning, and what problem does this project solve?**
   It learns to imitate recorded human actions. Here the CNN maps a simulator road image to the steering command demonstrated by a driver. It is only a steering predictor, not a complete driving system.

2. **Why is this supervised learning and regression?**
   Each image has a known steering label, providing supervision. Steering is a continuous number rather than a category, so the network uses regression with one linear output and MSE loss.

3. **Why choose a lightweight CNN?**
   Convolutions learn local visual patterns such as road edges. Our three-convolution model has 21,329 parameters, making CPU training and inference practical. `src/model.py` implements the agreed architecture.

4. **Which dataset did you actually use?**
   A genuine 300-frame consecutive excerpt from `zaynena/selfdriving-car-simulator`, with original center images and steering labels. The source log contains 32,443 rows. The project does not claim to have trained on the complete Kaggle dataset. Provenance is saved with the raw data.

5. **What are the inputs and target, and what are their units?**
   Input is a normalized RGB array of shape `(80, 160, 3)`. Target is the original continuous steering command in native simulator units, not degrees. Left/right labels remain disabled because the dataset-specific sign convention has not been independently verified.

6. **How do you validate the data?**
   `src/data_loader.py` checks CSV columns, finite numeric labels, missing paths, ambiguous filename recovery, readable images, repeated paths and byte-identical copies. Invalid rows are counted with reasons. Byte hashing does not identify every re-encoded or visually similar frame.

7. **Why split into training, validation and test sets?**
   Training fits weights, validation selects checkpoints and early stopping, and the fixed test set evaluates the selected model. We retained 205/35/40 records and excluded 20 boundary frames. Requested proportions are roughly 70/15/15 before gap exclusions.

8. **What is data leakage, and why not randomly split video frames?**
   Leakage gives training access to information that should belong to evaluation. Neighboring frames can be nearly identical, so random row splits can overstate performance. The code groups real sessions when provided, otherwise uses chronological blocks with boundary gaps. One short drive remains correlated despite gaps.

9. **Why does steering imbalance matter?**
   Many nearly straight examples can reward nearly constant outputs. The excerpt contains no positive original labels and 70.33% near-zero labels. A constant baseline is therefore essential; low average error alone does not establish useful turning behavior.

10. **Why crop and resize?**
    Cropping removes much of the sky/dashboard, concentrating on the road. Top 35% and bottom 10% are removed. Resizing to 160 x 80 reduces computation and gives every batch a consistent shape. `src/preprocess.py` implements both.

11. **What is normalization, and how is consistency preserved?**
    Pixels become float32 and are divided by 255 once to lie in `[0, 1]`. Training and inference use the same function. Model metadata stores crop/shape/normalization settings, and `load_trained_model` checks that contract before prediction.

12. **What augmentation is used, and why reverse the steering sign when flipping?**
    Training mirrors images with probability 0.5 and adjusts brightness by 0.8–1.2. Mirroring reverses the geometric turn, so the target sign must reverse too. Brightness leaves the target unchanged. Validation and test images are never augmented; mirrored examples are not independent new drives.

13. **What does a convolutional layer do?**
    It slides learned filters over image regions to detect patterns. The network combines simple features into more useful representations. Stride 2 also reduces spatial size. Our filters are 16, 32 and 48 in three layers.

14. **What is ReLU?**
    ReLU outputs `max(0, x)`. This nonlinearity lets multiple layers represent more than one linear transformation. It is used in convolutional layers and the 32-unit dense layer.

15. **What is pooling, and why global average pooling?**
    Pooling summarizes spatial information. Global average pooling reduces each final feature map to its mean, greatly reducing dense-layer parameters compared with flattening all spatial positions. This keeps the model small, although spatial detail is lost.

16. **What are dense layers, dropout and the linear output for?**
    Dense layers combine features. Dropout randomly removes 20% of activations during training to discourage dependence on a few features; inference turns it off. One linear output produces a continuous number without restricting it to classification probabilities.

17. **What are epoch and batch size?**
    An epoch visits the training records once, with new random augmentation. Batch size is how many examples contribute to one weight update. Defaults are batch 32 and up to 20 epochs. The image generator avoids holding the full image dataset in memory.

18. **What are Adam, learning rate and MSE?**
    Adam updates weights using gradient estimates and adaptive step sizes. Learning rate controls update size; default is 0.001. MSE averages squared prediction errors and penalizes large mistakes. These choices are in `src/model.py` and `src/train.py`.

19. **What are overfitting and early stopping?**
    Overfitting is learning training-specific patterns that do not generalize. Early stopping monitors validation loss and stops after four epochs without improvement. The saved checkpoint is the minimum-validation-loss model, not necessarily the last epoch. Training/validation curves use different augmentation/dropout behavior, so their absolute gap requires care.

20. **What are MAE and RMSE?**
    MAE averages absolute prediction errors. RMSE takes the square root of average squared errors, so large errors count more. Both have the same units as steering. `src/evaluate.py` computes them with NumPy and rejects invalid inputs.

21. **What is the mean-steering baseline, and how do you avoid leakage in it?**
    It predicts one constant: the mean of original training labels. That is about -0.107073 for this run. Test labels never determine this constant. CNN and baseline are measured on exactly the same test images.

22. **What did the real experiment show?**
    CPU training stopped after five epochs; epoch 1 had best validation MSE about 0.018133. On 40 test frames, CNN MAE/RMSE are about 0.082928/0.167125; baseline values are 0.134244/0.161830. Thus average absolute error improved, but squared-error sensitivity worsened. This is not a clean win or proof of learned driving. Read `RESULTS.md` for interpretation.

23. **What is Streamlit, and how does upload prediction work?**
    Streamlit builds an interactive Python dashboard. Upload bytes become a Pillow RGB image, pass through saved preprocessing, gain a batch dimension, and enter the cached trained CNN. `predict_steering` returns one finite float. The UI displays that value and a numeric gauge, never a label disguised as a prediction.

24. **How did you test reproducibility and the full workflow?**
    Fixed splits/seeds, hashes, runtime versions and saved settings make runs auditable. Pytest checks loading, preprocessing, split guards, model shape, save/reload, metrics and tiny end-to-end training. Streamlit tests exercise sample prediction, valid uploads and corrupt uploads. Synthetic fixture metrics stay in temporary folders; project reports use real images. Hardware/runtime changes can still alter exact floating-point results.

25. **Can this drive a real car, and what is the future scope?**
    No. It has no obstacle detection, traffic-signal recognition, braking, speed control or guarantee of safe navigation. The current excerpt is short, imbalanced and correlated. Future work should collect diverse independent drives and use validation-guided improvements, followed by a new independent final evaluation. Colab support provides optional training compute but does not solve data quality.
