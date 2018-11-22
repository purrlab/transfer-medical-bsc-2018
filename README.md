__Wat ga ik onderzoeken__
Een systimatische aanpak met betrekking tot transfer learning. dit houdt in dat ik het zelfde model en techniek gebruik om verschillende data sets te testen. Later zullen modelen veranderen tegenover zelfde data en zelfde techniek. Uit eindelijk kan er gekeken worden naar verschillende technieken. 

__Hoofdvraag__
Wat is de invloed van de pre trrain data set op de test data set.

Sub: Is er en samen hang tussen resultaat/nauwkeurigheid en pre tain/test data grootte.
Sub: Is er en samen hang tussen resultaat/nauwkeurigheid en pre tain/test data classificatie aantal.
Sub: Is er en samen hang tussen resultaat/nauwkeurigheid en pre tain val_loss.

__Idealiter: "Eigelijk een paar parameters vinden die er voor zorgen dat jij kan zeggen dat je transferlearing model een goede match is."__

__BEP__
Met Verschillende datasets en varianten daarvan, transfer learning toepassen op VGG16 met feature extraction. Deze resultaten dan vergelijken tegenover gekozen parameters zoals lengte data set etc. 
Eerst zullen er 2 korte experimenten plaats vinden. Dan zullen de functies gekozen worden en gespecialiseerd worden, tevens een run script.

__Progameer doelen__
Hierin zal omschreven staan wat er  zal worden geprogrammeerd in python. Met de versies komt er meer diepgang in.
Doel
Het doel van het script klinkt als volgt:
“Het script moet overzichtelijk en gelijk zijn voor verschillende inputs en technieken, zodat er een systematische aanpak wordt gehanteerd. Het moet dus uit functies bestaan die meerdere parameters bevatten zodat je er veel mee kan spelen. Tevens moet er in de gaten gehouden worden wat voor versie het is en wat er gebeurt.”
Sub doelen
We kunnen de doelen verdelen in Lange termijn en Korte termijn. Korte termijn houd vooral in dat het z.s.m. moet werken en nodig is voor resultaat. Lange termijn zijn uitbreidingen. 
Voor dat de Functie volwassen is zal er een junior function zijn. Een die werk, maar simpel is. Erna komt een senior function. Hierdoor kan voordat er veel tijd in worden gestopt gekeken worden of het de gewenste resultaten geeft.
Korte termijn doel één 
-	Functie die data in laad en verdeling maakt in Test en Train
-	Functie die data bewerkt voor keras (pre-processing)
-	Functie die tussen data kan kiezen
-	Functie die (VGG16) kan maken en manipuleren.
-	Functie die SVM kan toepassen op de feature vector
-	Functie die nauwkeurigheid meet d.m.v. test set
-	Functie die visuele hulp bied om te beoordelen of alles werkt
-	(Test functie die checkt of het script werkt, micro data set)
-	Script die alles samen voegt met veel uitleg.
 
 (https://www.google.nl/url?sa=i&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwjQ0oCNut7eAhUEZ1AKHY8AC4AQjRx6BAgBEAU&url=http%3A%2F%2Fadilmoujahid.com%2Fposts%2F2016%2F06%2Fintroduction-deep-learning-python-caffe%2F&psig=AOvVaw3suGGd-R1WsNKfN0CMAVHp&ust=1542648084290066)
Figuur 1. Data flow of transfer learning using feature extraction.  

Lange termijn doelen
-	visualisatie voor resultaten, grafieken ect.
-	Meerdere CNN’s
-	Meer technieken  fine tuning, DNN, verschillende diepte.
-	GUI  tkinter, Flask, Django
Side notes:
-	Data komt van Kaggle of veronika’s paper.
-	CNN en DNN komen van Keras
-	SVM wordt gedaan met de sklearn libary.
-	Test en train dataset worden ten alletijden gescheiden.

