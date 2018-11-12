# transfer-medical-bsc-2018
Systematische aanpak transfer-learning

Ik ga onderzoek doen naar de verschillen tussen resultaten van verschillende data sets voor transfer-learning. Tevens kijk ik naar meerdere technieken die daar voor gebruikt kunnen worden.

Als uitbreiding zou ik nog meerdere netwerken gebruiken. Ik zal beginnen met VGG16:
“The runner-up at the ILSVRC 2014 competition is dubbed VGGNet by the community and was developed by Simonyan and Zisserman . VGGNet consists of 16 convolutional layers and is very appealing because of its very uniform architecture. Similar to AlexNet, only 3x3 convolutions, but lots of filters. Trained on 4 GPUs for 2–3 weeks. It is currently the most preferred choice in the community for extracting features from images. The weight configuration of the VGGNet is publicly available and has been used in many other applications and challenges as a baseline feature extractor. However, VGGNet consists of 138 million parameters, which can be a bit challenging to handle.”

Uit eigen onderzoek bleek CaffeNet14, DeCAF, AlexNet en VGG16 het meest voorkomend. Hiervan is VGG16 het best presterende en ook het simpelst. Dit tweede zorg voor een soepel verloop en veel mogelijkheden. Wat ook veel voor kwam was dat mensen een eigen CNN maakte, deze waren klein en simpel. Neem aan dat dit was voor het beperken van rekenkracht.

De technieken die ik bekijk zijn fine-tuning en feature Extraction. Beide technieken kunnen op twee verschillende manieren gedaan worden.
Fine-tuning, Alle parameters opnieuw trainen of een paar parameters opnieuw leren, d.w.z. paar lagen zijn dan bevroren. Beide technieken hebben misschien nog een paar output lagen nodig.
Feature Extraction, Je knipt het CNN aan het eind of begin, en maakt er een 4096 vector van. Hier pas je een SVM op toe voor classificatie.

De data sets zijn ook te verdelen in Medical en non-Medical. Later kan naar andere patronen tussen pre-train en test gekeken worden.
