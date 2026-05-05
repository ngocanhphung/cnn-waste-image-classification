# cnn-waste-image-classification
CNN Waste Image Classification — Automated Sorting System

Dataset: 2.8K waste images of 6 categories - Plastic, Glass, Cardboard, Paper, Vegetation, Metal

Method: Custom CNNs · Transfer learning (MobileNetV2)

numpy · pandas · matplotlib · seaborn · scikit-learn · tensorflow / keras (including ImageDataGenerator, MobileNetV2)
_______
Goal: Sorting waste at scale into the correct categories without manual labour.

Method: Systematically experiment with different CNN architecture and augmentation level.

Outcome: Optimal solution is transfer learning to MobileNetV2 with fine-tuning applied to the last 30 layers.

Findings: 
1. By first training only the classification head and then fine-tuning the last 30 backbone layers, the model adapts ImageNet-learned features to waste features — achieving better generalisation than all custom CNNs at lower parameter cost.
2. Plastic is the most ambiguous category and frequently confused with Glass and Cardboard.

Limitations:
1. Model trained on clean, 1-item images, simple backgrounds, and might underperform when fed with images of waste in reality (wet, torn, contaminated items).
2. Risk of misclassifying Glass with Plastic: contaminating glass recycling streams with plastic.

Business Recommendations:
1. Collect more Plastic images — especially opaque, dark-coloured, and film-type plastics for diversity in training data
2. Deploy with confidence threshold — require human intervention when Maximum Softmax Probability < 0.75
