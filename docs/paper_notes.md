# Research Notes: Automated Crater Categorization via Multi-Stage CNNs

## Abstract
Automated crater detection is critical for understanding planetary history and safe landing site selection. CraterNet-X proposes a two-stage approach that decouples detection (localization) from classification (size quantification). This separation allows for specialized optimization of both stages, resulting in a system capable of handling varying lighting and terrain conditions on the lunar surface.

## 1. Introduction
The Moon's surface is covered with millions of craters ranging from centimeters to hundreds of kilometers in diameter. Manual mapping is no longer feasible given the petabytes of high-resolution data from various lunar orbiters.

## 2. Related Work
Most existing models use single-stage detectors like Faster R-CNN or SSD. While efficient, these often struggle with precise size categorization because the classification is done on limited feature maps. CraterNet-X improves upon this by using full-resolution crops for the classification stage.

## 3. Proposed Architecture (CraterNet-X)
The system architecture follows a "Narrowing Down" philosophy:
1. **Localization**: Use YOLOv8 to generate high-recall proposals.
2. **Specialization**: Use ResNet-50 on high-resolution crops to extract fine-grained features related to crater rims and shadows, which are key indicators of physical depth and radius.

## 4. Key Findings
- **Threshold Sensitivity**: The system is highly sensitive to the 0.05 normalized size threshold.
- **Lighting Invariance**: Training with Albumentations (random brightness/contrast) drastically improved the system's performance on the Moon's poles (shadow-heavy regions).

## 5. Conclusion & Future Work
Future work will focus on integrating **Mars (MRO)** imagery to evaluate the cross-domain generalizability of the CraterNet-X feature extractors.
