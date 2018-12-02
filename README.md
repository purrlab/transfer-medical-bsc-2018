__Transfer learning__
Dankzij transfer learning kan je met een kleine data set toch goede resultaten krijgen uit je deep learing model, voorheen was dit niet mogelijk en dacht men dat je grote data sets nodig hebt voor goede resultaten.

Transferlearning kan op twee manieren:
* Fine tuning: Je traint over het model, of een deel van het model heen. Soms vereist dit dat je er 2 of 3 lagen aan toe moet voegen om hem op je classes te kunnen aanpassen.
* Featurising: Je gebruikt de uitkomst of een gekozen laag, meestal platte laag, als een vector voor je SVM. Zo combineer je als waren twee learning algoritmes.

transfer learning wordt al veel toegepast in de medische beeldverwerking, vaak wordt hier imagenet als pre train model gebruikt of een andere medische data set. Het model dat het vaakst voorkomen zijn VGG16 of Caffeenet, de tweede komt vooral voor in papers omdat deze heel simpel is en minder zware hardware nodig heeft. 

Vaak wordt echter heel nauw naar 1 techniek of data set gekeken, er mist als het ware nog een systematische aanpak. Met namen tussen de verschillen can medische en niet medische data sets

__Wat ga ik onderzoeken__
Een systimatische aanpak met betrekking tot transfer learning. dit houdt in dat ik het zelfde model en techniek gebruik om verschillende data sets te testen. Met name het verschil tussen medische en niet medische data sets

Later zullen modelen veranderen tegenover zelfde data en zelfde techniek. Uit eindelijk kan er gekeken worden naar verschillende technieken. 

__Hoofdvraag__
Wat is de invloed van de pre trrain data set op de test data set.

Sub: Is er en samen hang tussen resultaat/nauwkeurigheid en pre tain/test data medisch en niet medisch.
Sub: Is er en samen hang tussen resultaat/nauwkeurigheid en pre tain/test data grootte.
Sub: Is er en samen hang tussen resultaat/nauwkeurigheid en pre tain/test data classificatie aantal.
Sub: Is er en samen hang tussen resultaat/nauwkeurigheid en pre tain val_loss.

__Idealiter: "Wat zijn eigenschappen van een goede pre train data set voor de transferlearning dataset."__

__BEP__
1. Bewijs dat het script een transferlearning taak kan verrichten. (week 1-2)
2. Reproduceer een vorig transferlearning project. (week 3)
3. Voor testen uit met verschillende parameters en plot deze (week 4-5)
4. Vergelijk data sets met elkaar en plot resultaat (week 6-7)
5. Breid uit. (week 8-10)
6. Verslag (week 6-10 )

__Het Sciript__

* AADataset: loads en generates data
* AAPretrain: Makes and trains models
* AATransferLearn: Preforms predictions and SVM
* AALogic: Has some experiments in it, calls fucntions in specific order
* AAAnalyseData: Will plot and Calculate AUC after models and svm has ran.
* LabnoteDoc: Creates files with used params and gotten results. Figure names are also saved.
* File: LAB: Contains the files created by LabnoteDoc
* Other files are "unsorted junk"




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

