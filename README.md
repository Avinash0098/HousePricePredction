
# House Price Prediction

A brief description of what this project does and who it's for


## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## Documentation

[Documentation](https://linktodocumentation)

# HousePricePredction
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
Load DataSet
train = pd.read_csv('Propert_Price_Train.csv')
test = pd.read_csv('Property_Price_Test.csv')

print("Shape of train: ", train.shape)
print("Shape of test: ", test.shape)
Shape of train:  (1459, 81)
Shape of test:  (1459, 80)
train.head(10)
Id	Building_Class	Zoning_Class	Lot_Extent	Lot_Size	Road_Type	Lane_Type	Property_Shape	Land_Outline	Utility_Type	...	Pool_Area	Pool_Quality	Fence_Quality	Miscellaneous_Feature	Miscellaneous_Value	Month_Sold	Year_Sold	Sale_Type	Sale_Condition	Sale_Price
0	1	60	RLD	65.0	8450	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	2	2008	WD	Normal	208500
1	2	20	RLD	80.0	9600	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	5	2007	WD	Normal	181500
2	3	60	RLD	68.0	11250	Paved	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	9	2008	WD	Normal	223500
3	4	70	RLD	60.0	9550	Paved	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	2	2006	WD	Abnorml	140000
4	5	60	RLD	84.0	14260	Paved	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	12	2008	WD	Normal	250000
5	6	50	RLD	85.0	14115	Paved	NaN	IR1	Lvl	AllPub	...	0	NaN	MnPrv	Shed	700	10	2009	WD	Normal	143000
6	7	20	RLD	75.0	10084	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	8	2007	WD	Normal	307000
7	8	60	RLD	NaN	10382	Paved	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	Shed	350	11	2009	WD	Normal	200000
8	9	50	RMD	51.0	6120	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	4	2008	WD	Abnorml	129900
9	10	190	RLD	50.0	7420	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	1	2008	WD	Normal	118000
10 rows × 81 columns

test.head(10)
Id	Building_Class	Zoning_Class	Lot_Extent	Lot_Size	Road_Type	Lane_Type	Property_Shape	Land_Outline	Utility_Type	...	Screen_Lobby_Area	Pool_Area	Pool_Quality	Fence_Quality	Miscellaneous_Feature	Miscellaneous_Value	Month_Sold	Year_Sold	Sale_Type	Sale_Condition
0	1461	20	RHD	80.0	16104.819760	Paved	NaN	Reg	Lvl	AllPub	...	120	0	NaN	MnPrv	NaN	0	6	2010	WD	Normal
1	1462	20	RLD	81.0	15639.150810	Paved	NaN	IR1	Lvl	AllPub	...	0	0	NaN	NaN	Gar2	12500	6	2010	WD	Normal
2	1463	60	RLD	74.0	3849.428920	Paved	NaN	IR1	Lvl	AllPub	...	0	0	NaN	MnPrv	NaN	0	3	2010	WD	Normal
3	1464	60	RLD	78.0	4955.447942	Paved	NaN	IR1	Lvl	AllPub	...	0	0	NaN	NaN	NaN	0	6	2010	WD	Normal
4	1465	120	RLD	43.0	3046.604942	Paved	NaN	IR1	HLS	AllPub	...	144	0	NaN	NaN	NaN	0	1	2010	WD	Normal
5	1466	60	RLD	75.0	10194.721200	Paved	NaN	IR1	Lvl	AllPub	...	0	0	NaN	NaN	NaN	0	4	2010	WD	Normal
6	1467	20	RLD	NaN	15033.338140	Paved	NaN	IR1	Lvl	AllPub	...	0	0	NaN	GdPrv	Shed	500	3	2010	WD	Normal
7	1468	60	RLD	63.0	12975.123110	Paved	NaN	IR1	Lvl	AllPub	...	0	0	NaN	NaN	NaN	0	5	2010	WD	Normal
8	1469	20	RLD	85.0	22272.588650	Paved	NaN	Reg	Lvl	AllPub	...	0	0	NaN	NaN	NaN	0	2	2010	WD	Normal
9	1470	20	RLD	70.0	11464.167120	Paved	NaN	Reg	Lvl	AllPub	...	0	0	NaN	MnPrv	NaN	0	4	2010	WD	Normal
10 rows × 80 columns

#Concat the train and test
df = pd.concat((train, test))
temp_df = df
print("Shape of df: ", df.shape)
Shape of df:  (2918, 81)
df.head(8)
Id	Building_Class	Zoning_Class	Lot_Extent	Lot_Size	Road_Type	Lane_Type	Property_Shape	Land_Outline	Utility_Type	...	Pool_Area	Pool_Quality	Fence_Quality	Miscellaneous_Feature	Miscellaneous_Value	Month_Sold	Year_Sold	Sale_Type	Sale_Condition	Sale_Price
0	1	60	RLD	65.0	8450.0	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	2	2008	WD	Normal	208500.0
1	2	20	RLD	80.0	9600.0	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	5	2007	WD	Normal	181500.0
2	3	60	RLD	68.0	11250.0	Paved	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	9	2008	WD	Normal	223500.0
3	4	70	RLD	60.0	9550.0	Paved	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	2	2006	WD	Abnorml	140000.0
4	5	60	RLD	84.0	14260.0	Paved	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	12	2008	WD	Normal	250000.0
5	6	50	RLD	85.0	14115.0	Paved	NaN	IR1	Lvl	AllPub	...	0	NaN	MnPrv	Shed	700	10	2009	WD	Normal	143000.0
6	7	20	RLD	75.0	10084.0	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	8	2007	WD	Normal	307000.0
7	8	60	RLD	NaN	10382.0	Paved	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	Shed	350	11	2009	WD	Normal	200000.0
8 rows × 81 columns

df.tail(8)
Id	Building_Class	Zoning_Class	Lot_Extent	Lot_Size	Road_Type	Lane_Type	Property_Shape	Land_Outline	Utility_Type	...	Pool_Area	Pool_Quality	Fence_Quality	Miscellaneous_Feature	Miscellaneous_Value	Month_Sold	Year_Sold	Sale_Type	Sale_Condition	Sale_Price
1451	2912	20	RLD	80.0	17795.736390	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	5	2006	WD	Normal	NaN
1452	2913	160	RMD	21.0	12104.290990	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	12	2006	WD	AbnoRMDl	NaN
1453	2914	160	RMD	21.0	20033.272180	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	GdPrv	NaN	0	6	2006	WD	NoRMDal	NaN
1454	2915	160	RMD	21.0	14584.838440	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	6	2006	WD	NoRMDal	NaN
1455	2916	160	RMD	21.0	8072.991379	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	4	2006	WD	AbnoRMDl	NaN
1456	2917	20	RLD	160.0	7367.775348	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	9	2006	WD	Abnorml	NaN
1457	2918	85	RLD	62.0	2203.135444	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	MnPrv	Shed	700	7	2006	WD	Normal	NaN
1458	2919	60	RLD	74.0	6253.431852	Paved	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	11	2006	WD	Normal	NaN
8 rows × 81 columns

Exploratory Data Analysis (EDA)
# Show the all columns
pd.set_option("display.max_columns", 2000)
pd.set_option("display.max_rows", 85)
df.head(8)
Id	Building_Class	Zoning_Class	Lot_Extent	Lot_Size	Road_Type	Lane_Type	Property_Shape	Land_Outline	Utility_Type	Lot_Configuration	Property_Slope	Neighborhood	Condition1	Condition2	House_Type	House_Design	Overall_Material	House_Condition	Construction_Year	Remodel_Year	Roof_Design	Roof_Quality	Exterior1st	Exterior2nd	Brick_Veneer_Type	Brick_Veneer_Area	Exterior_Material	Exterior_Condition	Foundation_Type	Basement_Height	Basement_Condition	Exposure_Level	BsmtFinType1	BsmtFinSF1	BsmtFinType2	BsmtFinSF2	BsmtUnfSF	Total_Basement_Area	Heating_Type	Heating_Quality	Air_Conditioning	Electrical_System	First_Floor_Area	Second_Floor_Area	LowQualFinSF	Grade_Living_Area	Underground_Full_Bathroom	Underground_Half_Bathroom	Full_Bathroom_Above_Grade	Half_Bathroom_Above_Grade	Bedroom_Above_Grade	Kitchen_Above_Grade	Kitchen_Quality	Rooms_Above_Grade	Functional_Rate	Fireplaces	Fireplace_Quality	Garage	Garage_Built_Year	Garage_Finish_Year	Garage_Size	Garage_Area	Garage_Quality	Garage_Condition	Pavedd_Drive	W_Deck_Area	Open_Lobby_Area	Enclosed_Lobby_Area	Three_Season_Lobby_Area	Screen_Lobby_Area	Pool_Area	Pool_Quality	Fence_Quality	Miscellaneous_Feature	Miscellaneous_Value	Month_Sold	Year_Sold	Sale_Type	Sale_Condition	Sale_Price
0	1	60	RLD	65.0	8450.0	Paved	NaN	Reg	Lvl	AllPub	I	GS	CollgCr	Norm	Norm	1Fam	2Story	7	5	2003	2003	Gable	SS	VinylSd	VinylSd	BrkFace	196.0	Gd	TA	PC	Gd	TA	No	GLQ	706.0	Unf	0.0	150.0	856.0	GasA	Ex	Y	SBrkr	856	854	0	1710	1.0	0.0	2	1	3	1	Gd	8	TF	0	NaN	Attchd	2003.0	RFn	2.0	1085.793744	TA	TA	Y	163.788080	69.596115	20.337934	0	0	0	NaN	NaN	NaN	0	2	2008	WD	Normal	208500.0
1	2	20	RLD	80.0	9600.0	Paved	NaN	Reg	Lvl	AllPub	FR2P	GS	Veenker	Feedr	Norm	1Fam	1Story	6	8	1976	1976	Gable	SS	MetalSd	MetalSd	None	0.0	TA	TA	CB	Gd	TA	Gd	ALQ	978.0	Unf	0.0	284.0	1262.0	GasA	Ex	Y	SBrkr	1262	0	0	1262	0.0	1.0	2	0	3	1	TA	6	TF	1	TA	Attchd	1976.0	RFn	2.0	196.316304	TA	TA	Y	198.900074	74.716033	15.039392	0	0	0	NaN	NaN	NaN	0	5	2007	WD	Normal	181500.0
2	3	60	RLD	68.0	11250.0	Paved	NaN	IR1	Lvl	AllPub	I	GS	CollgCr	Norm	Norm	1Fam	2Story	7	5	2001	2002	Gable	SS	VinylSd	VinylSd	BrkFace	162.0	Gd	TA	PC	Gd	TA	Mn	GLQ	486.0	Unf	0.0	434.0	920.0	GasA	Ex	Y	SBrkr	920	866	0	1786	1.0	0.0	2	1	3	1	Gd	6	TF	1	TA	Attchd	2001.0	RFn	2.0	218.068403	TA	TA	Y	26.127533	32.085268	-46.232198	0	0	0	NaN	NaN	NaN	0	9	2008	WD	Normal	223500.0
3	4	70	RLD	60.0	9550.0	Paved	NaN	IR1	Lvl	AllPub	C	GS	Crawfor	Norm	Norm	1Fam	2Story	7	5	1915	1970	Gable	SS	Wd Sdng	Wd Shng	None	0.0	TA	TA	BT	TA	Gd	No	ALQ	216.0	Unf	0.0	540.0	756.0	GasA	Gd	Y	SBrkr	961	756	0	1717	1.0	0.0	1	0	3	1	Gd	7	TF	1	Gd	Detchd	1998.0	Unf	3.0	696.996439	TA	TA	Y	46.948018	40.181415	60.921821	0	0	0	NaN	NaN	NaN	0	2	2006	WD	Abnorml	140000.0
4	5	60	RLD	84.0	14260.0	Paved	NaN	IR1	Lvl	AllPub	FR2P	GS	NoRidge	Norm	Norm	1Fam	2Story	8	5	2000	2000	Gable	SS	VinylSd	VinylSd	BrkFace	350.0	Gd	TA	PC	Gd	TA	Av	GLQ	655.0	Unf	0.0	490.0	1145.0	GasA	Ex	Y	SBrkr	1145	1053	0	2198	1.0	0.0	2	1	4	1	Gd	9	TF	1	TA	Attchd	2000.0	RFn	3.0	568.859882	TA	TA	Y	-10.626105	20.755323	21.788818	0	0	0	NaN	NaN	NaN	0	12	2008	WD	Normal	250000.0
5	6	50	RLD	85.0	14115.0	Paved	NaN	IR1	Lvl	AllPub	I	GS	Mitchel	Norm	Norm	1Fam	1.5Fin	5	5	1993	1995	Gable	SS	VinylSd	VinylSd	None	0.0	TA	TA	W	Gd	TA	No	GLQ	732.0	Unf	0.0	64.0	796.0	GasA	Ex	Y	SBrkr	796	566	0	1362	1.0	0.0	1	1	1	1	TA	5	TF	0	NaN	Attchd	1993.0	Unf	2.0	703.481359	TA	TA	Y	0.621402	36.740335	70.350362	320	0	0	NaN	MnPrv	Shed	700	10	2009	WD	Normal	143000.0
6	7	20	RLD	75.0	10084.0	Paved	NaN	Reg	Lvl	AllPub	I	GS	Somerst	Norm	Norm	1Fam	1Story	8	5	2004	2005	Gable	SS	VinylSd	VinylSd	Stone	186.0	Gd	TA	PC	Ex	TA	Av	GLQ	1369.0	Unf	0.0	317.0	1686.0	GasA	Ex	Y	SBrkr	1694	0	0	1694	1.0	0.0	2	0	3	1	Gd	7	TF	1	Gd	Attchd	2004.0	RFn	2.0	555.415694	TA	TA	Y	39.047177	118.613457	-7.064622	0	0	0	NaN	NaN	NaN	0	8	2007	WD	Normal	307000.0
7	8	60	RLD	NaN	10382.0	Paved	NaN	IR1	Lvl	AllPub	C	GS	NWAmes	PosN	Norm	1Fam	2Story	7	6	1973	1973	Gable	SS	HdBoard	HdBoard	Stone	240.0	TA	TA	CB	Gd	TA	Mn	ALQ	859.0	BLQ	32.0	216.0	1107.0	GasA	Ex	Y	SBrkr	1107	983	0	2090	1.0	0.0	2	1	3	1	TA	7	TF	2	TA	Attchd	1973.0	RFn	2.0	737.632993	TA	TA	Y	201.101046	150.621507	76.923944	0	0	0	NaN	NaN	Shed	350	11	2009	WD	Normal	200000.0
df.tail(8)
Id	Building_Class	Zoning_Class	Lot_Extent	Lot_Size	Road_Type	Lane_Type	Property_Shape	Land_Outline	Utility_Type	Lot_Configuration	Property_Slope	Neighborhood	Condition1	Condition2	House_Type	House_Design	Overall_Material	House_Condition	Construction_Year	Remodel_Year	Roof_Design	Roof_Quality	Exterior1st	Exterior2nd	Brick_Veneer_Type	Brick_Veneer_Area	Exterior_Material	Exterior_Condition	Foundation_Type	Basement_Height	Basement_Condition	Exposure_Level	BsmtFinType1	BsmtFinSF1	BsmtFinType2	BsmtFinSF2	BsmtUnfSF	Total_Basement_Area	Heating_Type	Heating_Quality	Air_Conditioning	Electrical_System	First_Floor_Area	Second_Floor_Area	LowQualFinSF	Grade_Living_Area	Underground_Full_Bathroom	Underground_Half_Bathroom	Full_Bathroom_Above_Grade	Half_Bathroom_Above_Grade	Bedroom_Above_Grade	Kitchen_Above_Grade	Kitchen_Quality	Rooms_Above_Grade	Functional_Rate	Fireplaces	Fireplace_Quality	Garage	Garage_Built_Year	Garage_Finish_Year	Garage_Size	Garage_Area	Garage_Quality	Garage_Condition	Pavedd_Drive	W_Deck_Area	Open_Lobby_Area	Enclosed_Lobby_Area	Three_Season_Lobby_Area	Screen_Lobby_Area	Pool_Area	Pool_Quality	Fence_Quality	Miscellaneous_Feature	Miscellaneous_Value	Month_Sold	Year_Sold	Sale_Type	Sale_Condition	Sale_Price
1451	2912	20	RLD	80.0	17795.736390	Paved	NaN	Reg	Lvl	AllPub	I	MS	Mitchel	Norm	Norm	1Fam	1Story	5	5	1969	1979	Gable	SS	Plywood	Plywood	BrkFace	194.0	TA	TA	PC	TA	TA	Av	Rec	119.0	BLQ	344.0	641.0	1104.0	GasA	Fa	Y	SBrkr	1360	0	0	1360	1.0	0.0	1	0	3	1	TA	8	TF	1	TA	Attchd	1969.0	RFn	1.0	336.0	TA	TA	Y	160.0	0.0	0.0	0	0	0	NaN	NaN	NaN	0	5	2006	WD	Normal	NaN
1452	2913	160	RMD	21.0	12104.290990	Paved	NaN	Reg	Lvl	AllPub	I	GS	MeadowV	NoRMD	NoRMD	Twnhs	2Story	4	5	1970	1970	Gable	SS	CemntBd	CmentBd	None	0.0	TA	TA	CB	TA	TA	No	Rec	408.0	Unf	0.0	138.0	546.0	GasA	TA	Y	SBrkr	546	546	0	1092	0.0	0.0	1	1	3	1	TA	5	TF	0	NaN	CarPort	1970.0	Unf	1.0	286.0	TA	TA	Y	0.0	0.0	0.0	0	0	0	NaN	NaN	NaN	0	12	2006	WD	AbnoRMDl	NaN
1453	2914	160	RMD	21.0	20033.272180	Paved	NaN	Reg	Lvl	AllPub	I	GS	MeadowV	NoRMD	NoRMD	Twnhs	2Story	4	5	1970	1970	Gable	SS	CemntBd	CmentBd	None	0.0	TA	TA	CB	TA	TA	No	Unf	0.0	Unf	0.0	546.0	546.0	GasA	TA	Y	SBrkr	546	546	0	1092	0.0	0.0	1	1	3	1	TA	5	TF	0	NaN	NaN	NaN	NaN	0.0	0.0	NaN	NaN	Y	0.0	34.0	0.0	0	0	0	NaN	GdPrv	NaN	0	6	2006	WD	NoRMDal	NaN
1454	2915	160	RMD	21.0	14584.838440	Paved	NaN	Reg	Lvl	AllPub	I	GS	MeadowV	NoRMD	NoRMD	Twnhs	2Story	4	7	1970	1970	Gable	SS	CemntBd	CmentBd	None	0.0	TA	TA	CB	TA	TA	No	Unf	0.0	Unf	0.0	546.0	546.0	GasA	Gd	Y	SBrkr	546	546	0	1092	0.0	0.0	1	1	3	1	TA	5	TF	0	NaN	NaN	NaN	NaN	0.0	0.0	NaN	NaN	Y	0.0	0.0	0.0	0	0	0	NaN	NaN	NaN	0	6	2006	WD	NoRMDal	NaN
1455	2916	160	RMD	21.0	8072.991379	Paved	NaN	Reg	Lvl	AllPub	I	GS	MeadowV	NoRMD	NoRMD	TwnhsE	2Story	4	5	1970	1970	Gable	SS	CemntBd	CmentBd	None	0.0	TA	TA	CB	TA	TA	No	Rec	252.0	Unf	0.0	294.0	546.0	GasA	TA	Y	SBrkr	546	546	0	1092	0.0	0.0	1	1	3	1	TA	6	TF	0	NaN	CarPort	1970.0	Unf	1.0	286.0	TA	TA	Y	0.0	24.0	0.0	0	0	0	NaN	NaN	NaN	0	4	2006	WD	AbnoRMDl	NaN
1456	2917	20	RLD	160.0	7367.775348	Paved	NaN	Reg	Lvl	AllPub	I	GS	Mitchel	Norm	Norm	1Fam	1Story	5	7	1960	1996	Gable	SS	VinylSd	VinylSd	None	0.0	TA	TA	CB	TA	TA	No	ALQ	1224.0	Unf	0.0	0.0	1224.0	GasA	Ex	Y	SBrkr	1224	0	0	1224	1.0	0.0	1	0	4	1	TA	7	TF	1	TA	Detchd	1960.0	Unf	2.0	576.0	TA	TA	Y	474.0	0.0	0.0	0	0	0	NaN	NaN	NaN	0	9	2006	WD	Abnorml	NaN
1457	2918	85	RLD	62.0	2203.135444	Paved	NaN	Reg	Lvl	AllPub	I	GS	Mitchel	Norm	Norm	1Fam	SFoyer	5	5	1992	1992	Gable	SS	HdBoard	Wd Shng	None	0.0	TA	TA	PC	Gd	TA	Av	GLQ	337.0	Unf	0.0	575.0	912.0	GasA	TA	Y	SBrkr	970	0	0	970	0.0	1.0	1	0	3	1	TA	6	TF	0	NaN	NaN	NaN	NaN	0.0	0.0	NaN	NaN	Y	80.0	32.0	0.0	0	0	0	NaN	MnPrv	Shed	700	7	2006	WD	Normal	NaN
1458	2919	60	RLD	74.0	6253.431852	Paved	NaN	Reg	Lvl	AllPub	I	MS	Mitchel	Norm	Norm	1Fam	2Story	7	5	1993	1994	Gable	SS	HdBoard	HdBoard	BrkFace	94.0	TA	TA	PC	Gd	TA	Av	LwQ	758.0	Unf	0.0	238.0	996.0	GasA	Ex	Y	SBrkr	996	1004	0	2000	0.0	0.0	2	1	3	1	TA	9	TF	1	TA	Attchd	1993.0	Fin	3.0	650.0	TA	TA	Y	190.0	48.0	0.0	0	0	0	NaN	NaN	NaN	0	11	2006	WD	Normal	NaN
df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 2918 entries, 0 to 1458
Data columns (total 81 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   Id                         2918 non-null   int64  
 1   Building_Class             2918 non-null   int64  
 2   Zoning_Class               2914 non-null   object 
 3   Lot_Extent                 2432 non-null   float64
 4   Lot_Size                   2918 non-null   float64
 5   Road_Type                  2918 non-null   object 
 6   Lane_Type                  198 non-null    object 
 7   Property_Shape             2918 non-null   object 
 8   Land_Outline               2918 non-null   object 
 9   Utility_Type               2916 non-null   object 
 10  Lot_Configuration          2918 non-null   object 
 11  Property_Slope             2918 non-null   object 
 12  Neighborhood               2918 non-null   object 
 13  Condition1                 2918 non-null   object 
 14  Condition2                 2918 non-null   object 
 15  House_Type                 2918 non-null   object 
 16  House_Design               2918 non-null   object 
 17  Overall_Material           2918 non-null   int64  
 18  House_Condition            2918 non-null   int64  
 19  Construction_Year          2918 non-null   int64  
 20  Remodel_Year               2918 non-null   int64  
 21  Roof_Design                2918 non-null   object 
 22  Roof_Quality               2918 non-null   object 
 23  Exterior1st                2917 non-null   object 
 24  Exterior2nd                2917 non-null   object 
 25  Brick_Veneer_Type          2894 non-null   object 
 26  Brick_Veneer_Area          2895 non-null   float64
 27  Exterior_Material          2918 non-null   object 
 28  Exterior_Condition         2918 non-null   object 
 29  Foundation_Type            2918 non-null   object 
 30  Basement_Height            2837 non-null   object 
 31  Basement_Condition         2836 non-null   object 
 32  Exposure_Level             2836 non-null   object 
 33  BsmtFinType1               2839 non-null   object 
 34  BsmtFinSF1                 2917 non-null   float64
 35  BsmtFinType2               2838 non-null   object 
 36  BsmtFinSF2                 2917 non-null   float64
 37  BsmtUnfSF                  2917 non-null   float64
 38  Total_Basement_Area        2917 non-null   float64
 39  Heating_Type               2918 non-null   object 
 40  Heating_Quality            2918 non-null   object 
 41  Air_Conditioning           2918 non-null   object 
 42  Electrical_System          2917 non-null   object 
 43  First_Floor_Area           2918 non-null   int64  
 44  Second_Floor_Area          2918 non-null   int64  
 45  LowQualFinSF               2918 non-null   int64  
 46  Grade_Living_Area          2918 non-null   int64  
 47  Underground_Full_Bathroom  2916 non-null   float64
 48  Underground_Half_Bathroom  2916 non-null   float64
 49  Full_Bathroom_Above_Grade  2918 non-null   int64  
 50  Half_Bathroom_Above_Grade  2918 non-null   int64  
 51  Bedroom_Above_Grade        2918 non-null   int64  
 52  Kitchen_Above_Grade        2918 non-null   int64  
 53  Kitchen_Quality            2917 non-null   object 
 54  Rooms_Above_Grade          2918 non-null   int64  
 55  Functional_Rate            2916 non-null   object 
 56  Fireplaces                 2918 non-null   int64  
 57  Fireplace_Quality          1499 non-null   object 
 58  Garage                     2761 non-null   object 
 59  Garage_Built_Year          2759 non-null   float64
 60  Garage_Finish_Year         2759 non-null   object 
 61  Garage_Size                2917 non-null   float64
 62  Garage_Area                2917 non-null   float64
 63  Garage_Quality             2759 non-null   object 
 64  Garage_Condition           2759 non-null   object 
 65  Pavedd_Drive               2918 non-null   object 
 66  W_Deck_Area                2918 non-null   float64
 67  Open_Lobby_Area            2918 non-null   float64
 68  Enclosed_Lobby_Area        2918 non-null   float64
 69  Three_Season_Lobby_Area    2918 non-null   int64  
 70  Screen_Lobby_Area          2918 non-null   int64  
 71  Pool_Area                  2918 non-null   int64  
 72  Pool_Quality               10 non-null     object 
 73  Fence_Quality              571 non-null    object 
 74  Miscellaneous_Feature      105 non-null    object 
 75  Miscellaneous_Value        2918 non-null   int64  
 76  Month_Sold                 2918 non-null   int64  
 77  Year_Sold                  2918 non-null   int64  
 78  Sale_Type                  2917 non-null   object 
 79  Sale_Condition             2918 non-null   object 
 80  Sale_Price                 1459 non-null   float64
dtypes: float64(16), int64(22), object(43)
memory usage: 1.8+ MB
df.describe()
Id	Building_Class	Lot_Extent	Lot_Size	Overall_Material	House_Condition	Construction_Year	Remodel_Year	Brick_Veneer_Area	BsmtFinSF1	BsmtFinSF2	BsmtUnfSF	Total_Basement_Area	First_Floor_Area	Second_Floor_Area	LowQualFinSF	Grade_Living_Area	Underground_Full_Bathroom	Underground_Half_Bathroom	Full_Bathroom_Above_Grade	Half_Bathroom_Above_Grade	Bedroom_Above_Grade	Kitchen_Above_Grade	Rooms_Above_Grade	Fireplaces	Garage_Built_Year	Garage_Size	Garage_Area	W_Deck_Area	Open_Lobby_Area	Enclosed_Lobby_Area	Three_Season_Lobby_Area	Screen_Lobby_Area	Pool_Area	Miscellaneous_Value	Month_Sold	Year_Sold	Sale_Price
count	2918.000000	2918.000000	2432.000000	2918.000000	2918.000000	2918.000000	2918.000000	2918.000000	2895.000000	2917.000000	2917.000000	2917.000000	2917.000000	2918.000000	2918.000000	2918.000000	2918.000000	2916.000000	2916.000000	2918.000000	2918.000000	2918.000000	2918.000000	2918.000000	2918.000000	2759.000000	2917.000000	2917.000000	2918.000000	2918.000000	2918.000000	2918.000000	2918.000000	2918.000000	2918.000000	2918.000000	2918.000000	1459.000000
mean	1460.000000	57.150446	69.303454	10194.634957	6.089445	5.564428	1971.314942	1984.271076	102.236615	441.290024	49.499829	560.917724	1051.707576	1159.548663	336.599040	4.696025	1500.843729	0.429698	0.061385	1.568197	0.380055	2.860178	1.044551	6.451679	0.597327	1978.118159	1.766884	471.851510	93.095222	48.062908	24.411772	2.603153	16.067855	2.252570	50.843386	6.213160	2007.792666	180944.102810
std	842.931492	42.519354	23.349420	7888.702911	1.410045	1.113292	30.296408	20.894880	179.355169	455.632103	169.176028	439.548616	440.825601	392.425265	428.729653	46.404695	506.117484	0.524719	0.245726	0.552964	0.502827	0.822830	0.214497	1.569626	0.646145	25.577701	0.761623	213.846684	126.257764	68.167925	64.346881	25.192440	56.193208	35.670034	567.498680	2.715224	1.315184	79464.918335
min	1.000000	20.000000	21.000000	-4265.104479	1.000000	1.000000	1872.000000	1950.000000	0.000000	0.000000	0.000000	0.000000	0.000000	334.000000	0.000000	0.000000	334.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	2.000000	0.000000	1895.000000	0.000000	-129.369350	-338.112031	-187.149958	-164.807386	0.000000	0.000000	0.000000	0.000000	1.000000	2006.000000	34900.000000
25%	730.250000	20.000000	59.000000	7134.500000	5.000000	5.000000	1953.250000	1965.000000	0.000000	0.000000	0.000000	220.000000	793.000000	876.000000	0.000000	0.000000	1126.000000	0.000000	0.000000	1.000000	0.000000	2.000000	1.000000	5.000000	0.000000	1960.000000	1.000000	323.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	4.000000	2007.000000	129950.000000
50%	1460.000000	50.000000	68.000000	9600.000000	6.000000	5.000000	1973.000000	1993.000000	0.000000	368.000000	0.000000	467.000000	989.000000	1082.000000	0.000000	0.000000	1444.000000	0.000000	0.000000	2.000000	0.000000	3.000000	1.000000	6.000000	1.000000	1979.000000	2.000000	477.000000	75.123594	36.000000	0.000000	0.000000	0.000000	0.000000	0.000000	6.000000	2008.000000	163000.000000
75%	2189.750000	70.000000	80.000000	12342.000000	7.000000	6.000000	2001.000000	2004.000000	164.000000	733.000000	0.000000	806.000000	1302.000000	1387.750000	704.000000	0.000000	1743.750000	1.000000	0.000000	2.000000	1.000000	3.000000	1.000000	7.000000	1.000000	2002.000000	2.000000	593.481992	173.183207	86.733331	50.325034	0.000000	0.000000	0.000000	0.000000	8.000000	2009.000000	214000.000000
max	2919.000000	190.000000	313.000000	215245.000000	10.000000	9.000000	2010.000000	2010.000000	1600.000000	5644.000000	1526.000000	2336.000000	6110.000000	5095.000000	2065.000000	1064.000000	5642.000000	3.000000	2.000000	4.000000	2.000000	8.000000	3.000000	15.000000	4.000000	2207.000000	5.000000	1488.000000	1424.000000	742.000000	1012.000000	508.000000	576.000000	800.000000	17000.000000	12.000000	2010.000000	755000.000000
df.select_dtypes(include=['int64', 'float64']).columns
Index(['Id', 'Building_Class', 'Lot_Extent', 'Lot_Size', 'Overall_Material',
       'House_Condition', 'Construction_Year', 'Remodel_Year',
       'Brick_Veneer_Area', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
       'Total_Basement_Area', 'First_Floor_Area', 'Second_Floor_Area',
       'LowQualFinSF', 'Grade_Living_Area', 'Underground_Full_Bathroom',
       'Underground_Half_Bathroom', 'Full_Bathroom_Above_Grade',
       'Half_Bathroom_Above_Grade', 'Bedroom_Above_Grade',
       'Kitchen_Above_Grade', 'Rooms_Above_Grade', 'Fireplaces',
       'Garage_Built_Year', 'Garage_Size', 'Garage_Area', 'W_Deck_Area',
       'Open_Lobby_Area', 'Enclosed_Lobby_Area', 'Three_Season_Lobby_Area',
       'Screen_Lobby_Area', 'Pool_Area', 'Miscellaneous_Value', 'Month_Sold',
       'Year_Sold', 'Sale_Price'],
      dtype='object')
df.select_dtypes(include=['object']).columns
Index(['Zoning_Class', 'Road_Type', 'Lane_Type', 'Property_Shape',
       'Land_Outline', 'Utility_Type', 'Lot_Configuration', 'Property_Slope',
       'Neighborhood', 'Condition1', 'Condition2', 'House_Type',
       'House_Design', 'Roof_Design', 'Roof_Quality', 'Exterior1st',
       'Exterior2nd', 'Brick_Veneer_Type', 'Exterior_Material',
       'Exterior_Condition', 'Foundation_Type', 'Basement_Height',
       'Basement_Condition', 'Exposure_Level', 'BsmtFinType1', 'BsmtFinType2',
       'Heating_Type', 'Heating_Quality', 'Air_Conditioning',
       'Electrical_System', 'Kitchen_Quality', 'Functional_Rate',
       'Fireplace_Quality', 'Garage', 'Garage_Finish_Year', 'Garage_Quality',
       'Garage_Condition', 'Pavedd_Drive', 'Pool_Quality', 'Fence_Quality',
       'Miscellaneous_Feature', 'Sale_Type', 'Sale_Condition'],
      dtype='object')
# To set the index as Id column
df = df.set_index("Id")
df.head(8)
Building_Class	Zoning_Class	Lot_Extent	Lot_Size	Road_Type	Lane_Type	Property_Shape	Land_Outline	Utility_Type	Lot_Configuration	Property_Slope	Neighborhood	Condition1	Condition2	House_Type	House_Design	Overall_Material	House_Condition	Construction_Year	Remodel_Year	Roof_Design	Roof_Quality	Exterior1st	Exterior2nd	Brick_Veneer_Type	Brick_Veneer_Area	Exterior_Material	Exterior_Condition	Foundation_Type	Basement_Height	Basement_Condition	Exposure_Level	BsmtFinType1	BsmtFinSF1	BsmtFinType2	BsmtFinSF2	BsmtUnfSF	Total_Basement_Area	Heating_Type	Heating_Quality	Air_Conditioning	Electrical_System	First_Floor_Area	Second_Floor_Area	LowQualFinSF	Grade_Living_Area	Underground_Full_Bathroom	Underground_Half_Bathroom	Full_Bathroom_Above_Grade	Half_Bathroom_Above_Grade	Bedroom_Above_Grade	Kitchen_Above_Grade	Kitchen_Quality	Rooms_Above_Grade	Functional_Rate	Fireplaces	Fireplace_Quality	Garage	Garage_Built_Year	Garage_Finish_Year	Garage_Size	Garage_Area	Garage_Quality	Garage_Condition	Pavedd_Drive	W_Deck_Area	Open_Lobby_Area	Enclosed_Lobby_Area	Three_Season_Lobby_Area	Screen_Lobby_Area	Pool_Area	Pool_Quality	Fence_Quality	Miscellaneous_Feature	Miscellaneous_Value	Month_Sold	Year_Sold	Sale_Type	Sale_Condition	Sale_Price
Id																																																																																
1	60	RLD	65.0	8450.0	Paved	NaN	Reg	Lvl	AllPub	I	GS	CollgCr	Norm	Norm	1Fam	2Story	7	5	2003	2003	Gable	SS	VinylSd	VinylSd	BrkFace	196.0	Gd	TA	PC	Gd	TA	No	GLQ	706.0	Unf	0.0	150.0	856.0	GasA	Ex	Y	SBrkr	856	854	0	1710	1.0	0.0	2	1	3	1	Gd	8	TF	0	NaN	Attchd	2003.0	RFn	2.0	1085.793744	TA	TA	Y	163.788080	69.596115	20.337934	0	0	0	NaN	NaN	NaN	0	2	2008	WD	Normal	208500.0
2	20	RLD	80.0	9600.0	Paved	NaN	Reg	Lvl	AllPub	FR2P	GS	Veenker	Feedr	Norm	1Fam	1Story	6	8	1976	1976	Gable	SS	MetalSd	MetalSd	None	0.0	TA	TA	CB	Gd	TA	Gd	ALQ	978.0	Unf	0.0	284.0	1262.0	GasA	Ex	Y	SBrkr	1262	0	0	1262	0.0	1.0	2	0	3	1	TA	6	TF	1	TA	Attchd	1976.0	RFn	2.0	196.316304	TA	TA	Y	198.900074	74.716033	15.039392	0	0	0	NaN	NaN	NaN	0	5	2007	WD	Normal	181500.0
3	60	RLD	68.0	11250.0	Paved	NaN	IR1	Lvl	AllPub	I	GS	CollgCr	Norm	Norm	1Fam	2Story	7	5	2001	2002	Gable	SS	VinylSd	VinylSd	BrkFace	162.0	Gd	TA	PC	Gd	TA	Mn	GLQ	486.0	Unf	0.0	434.0	920.0	GasA	Ex	Y	SBrkr	920	866	0	1786	1.0	0.0	2	1	3	1	Gd	6	TF	1	TA	Attchd	2001.0	RFn	2.0	218.068403	TA	TA	Y	26.127533	32.085268	-46.232198	0	0	0	NaN	NaN	NaN	0	9	2008	WD	Normal	223500.0
4	70	RLD	60.0	9550.0	Paved	NaN	IR1	Lvl	AllPub	C	GS	Crawfor	Norm	Norm	1Fam	2Story	7	5	1915	1970	Gable	SS	Wd Sdng	Wd Shng	None	0.0	TA	TA	BT	TA	Gd	No	ALQ	216.0	Unf	0.0	540.0	756.0	GasA	Gd	Y	SBrkr	961	756	0	1717	1.0	0.0	1	0	3	1	Gd	7	TF	1	Gd	Detchd	1998.0	Unf	3.0	696.996439	TA	TA	Y	46.948018	40.181415	60.921821	0	0	0	NaN	NaN	NaN	0	2	2006	WD	Abnorml	140000.0
5	60	RLD	84.0	14260.0	Paved	NaN	IR1	Lvl	AllPub	FR2P	GS	NoRidge	Norm	Norm	1Fam	2Story	8	5	2000	2000	Gable	SS	VinylSd	VinylSd	BrkFace	350.0	Gd	TA	PC	Gd	TA	Av	GLQ	655.0	Unf	0.0	490.0	1145.0	GasA	Ex	Y	SBrkr	1145	1053	0	2198	1.0	0.0	2	1	4	1	Gd	9	TF	1	TA	Attchd	2000.0	RFn	3.0	568.859882	TA	TA	Y	-10.626105	20.755323	21.788818	0	0	0	NaN	NaN	NaN	0	12	2008	WD	Normal	250000.0
6	50	RLD	85.0	14115.0	Paved	NaN	IR1	Lvl	AllPub	I	GS	Mitchel	Norm	Norm	1Fam	1.5Fin	5	5	1993	1995	Gable	SS	VinylSd	VinylSd	None	0.0	TA	TA	W	Gd	TA	No	GLQ	732.0	Unf	0.0	64.0	796.0	GasA	Ex	Y	SBrkr	796	566	0	1362	1.0	0.0	1	1	1	1	TA	5	TF	0	NaN	Attchd	1993.0	Unf	2.0	703.481359	TA	TA	Y	0.621402	36.740335	70.350362	320	0	0	NaN	MnPrv	Shed	700	10	2009	WD	Normal	143000.0
7	20	RLD	75.0	10084.0	Paved	NaN	Reg	Lvl	AllPub	I	GS	Somerst	Norm	Norm	1Fam	1Story	8	5	2004	2005	Gable	SS	VinylSd	VinylSd	Stone	186.0	Gd	TA	PC	Ex	TA	Av	GLQ	1369.0	Unf	0.0	317.0	1686.0	GasA	Ex	Y	SBrkr	1694	0	0	1694	1.0	0.0	2	0	3	1	Gd	7	TF	1	Gd	Attchd	2004.0	RFn	2.0	555.415694	TA	TA	Y	39.047177	118.613457	-7.064622	0	0	0	NaN	NaN	NaN	0	8	2007	WD	Normal	307000.0
8	60	RLD	NaN	10382.0	Paved	NaN	IR1	Lvl	AllPub	C	GS	NWAmes	PosN	Norm	1Fam	2Story	7	6	1973	1973	Gable	SS	HdBoard	HdBoard	Stone	240.0	TA	TA	CB	Gd	TA	Mn	ALQ	859.0	BLQ	32.0	216.0	1107.0	GasA	Ex	Y	SBrkr	1107	983	0	2090	1.0	0.0	2	1	3	1	TA	7	TF	2	TA	Attchd	1973.0	RFn	2.0	737.632993	TA	TA	Y	201.101046	150.621507	76.923944	0	0	0	NaN	NaN	Shed	350	11	2009	WD	Normal	200000.0
# Null values using heatmap

plt.figure(figsize=(16,9))
sns.heatmap(df.isnull())
<AxesSubplot:ylabel='Id'>

#Percentages of null value

null_percent = df.isnull().sum()/df.shape[0]*100
null_percent
Building_Class                0.000000
Zoning_Class                  0.137080
Lot_Extent                   16.655243
Lot_Size                      0.000000
Road_Type                     0.000000
Lane_Type                    93.214531
Property_Shape                0.000000
Land_Outline                  0.000000
Utility_Type                  0.068540
Lot_Configuration             0.000000
Property_Slope                0.000000
Neighborhood                  0.000000
Condition1                    0.000000
Condition2                    0.000000
House_Type                    0.000000
House_Design                  0.000000
Overall_Material              0.000000
House_Condition               0.000000
Construction_Year             0.000000
Remodel_Year                  0.000000
Roof_Design                   0.000000
Roof_Quality                  0.000000
Exterior1st                   0.034270
Exterior2nd                   0.034270
Brick_Veneer_Type             0.822481
Brick_Veneer_Area             0.788211
Exterior_Material             0.000000
Exterior_Condition            0.000000
Foundation_Type               0.000000
Basement_Height               2.775874
Basement_Condition            2.810144
Exposure_Level                2.810144
BsmtFinType1                  2.707334
BsmtFinSF1                    0.034270
BsmtFinType2                  2.741604
BsmtFinSF2                    0.034270
BsmtUnfSF                     0.034270
Total_Basement_Area           0.034270
Heating_Type                  0.000000
Heating_Quality               0.000000
Air_Conditioning              0.000000
Electrical_System             0.034270
First_Floor_Area              0.000000
Second_Floor_Area             0.000000
LowQualFinSF                  0.000000
Grade_Living_Area             0.000000
Underground_Full_Bathroom     0.068540
Underground_Half_Bathroom     0.068540
Full_Bathroom_Above_Grade     0.000000
Half_Bathroom_Above_Grade     0.000000
Bedroom_Above_Grade           0.000000
Kitchen_Above_Grade           0.000000
Kitchen_Quality               0.034270
Rooms_Above_Grade             0.000000
Functional_Rate               0.068540
Fireplaces                    0.000000
Fireplace_Quality            48.629198
Garage                        5.380398
Garage_Built_Year             5.448938
Garage_Finish_Year            5.448938
Garage_Size                   0.034270
Garage_Area                   0.034270
Garage_Quality                5.448938
Garage_Condition              5.448938
Pavedd_Drive                  0.000000
W_Deck_Area                   0.000000
Open_Lobby_Area               0.000000
Enclosed_Lobby_Area           0.000000
Three_Season_Lobby_Area       0.000000
Screen_Lobby_Area             0.000000
Pool_Area                     0.000000
Pool_Quality                 99.657300
Fence_Quality                80.431803
Miscellaneous_Feature        96.401645
Miscellaneous_Value           0.000000
Month_Sold                    0.000000
Year_Sold                     0.000000
Sale_Type                     0.034270
Sale_Condition                0.000000
Sale_Price                   50.000000
dtype: float64
col_for_drop = null_percent[null_percent > 20].keys() # if the null value % 20 or > 20 so need to drop it
# drop the columns

df = df.drop(col_for_drop, "columns")
df.shape
(2918, 74)
# find the unique value count
for i in df.columns:
    print(i + "\t" + str(len(df[i].unique())))
Building_Class	16
Zoning_Class	6
Lot_Extent	129
Lot_Size	2532
Road_Type	2
Property_Shape	4
Land_Outline	4
Utility_Type	3
Lot_Configuration	5
Property_Slope	3
Neighborhood	25
Condition1	10
Condition2	9
House_Type	5
House_Design	8
Overall_Material	10
House_Condition	9
Construction_Year	118
Remodel_Year	61
Roof_Design	6
Roof_Quality	8
Exterior1st	17
Exterior2nd	17
Brick_Veneer_Type	5
Brick_Veneer_Area	445
Exterior_Material	4
Exterior_Condition	5
Foundation_Type	6
Basement_Height	5
Basement_Condition	5
Exposure_Level	5
BsmtFinType1	7
BsmtFinSF1	991
BsmtFinType2	7
BsmtFinSF2	273
BsmtUnfSF	1136
Total_Basement_Area	1059
Heating_Type	6
Heating_Quality	5
Air_Conditioning	2
Electrical_System	6
First_Floor_Area	1083
Second_Floor_Area	635
LowQualFinSF	36
Grade_Living_Area	1292
Underground_Full_Bathroom	5
Underground_Half_Bathroom	4
Full_Bathroom_Above_Grade	5
Half_Bathroom_Above_Grade	3
Bedroom_Above_Grade	8
Kitchen_Above_Grade	4
Kitchen_Quality	5
Rooms_Above_Grade	14
Functional_Rate	11
Fireplaces	5
Garage	8
Garage_Built_Year	104
Garage_Finish_Year	4
Garage_Size	7
Garage_Area	1919
Garage_Quality	6
Garage_Condition	6
Pavedd_Drive	3
W_Deck_Area	1722
Open_Lobby_Area	1662
Enclosed_Lobby_Area	1590
Three_Season_Lobby_Area	31
Screen_Lobby_Area	121
Pool_Area	14
Miscellaneous_Value	38
Month_Sold	12
Year_Sold	5
Sale_Type	10
Sale_Condition	8
# find unique values of each column
for i in df.columns:
    print("Unique value of:>>> {} ({})\n{}\n".format(i, len(df[i].unique()), df[i].unique()))
Unique value of:>>> Building_Class (16)
[ 60  20  70  50 190  45  90 120  30  85  80 160  75 180  40 150]

Unique value of:>>> Zoning_Class (6)
['RLD' 'RMD' 'Commer' 'FVR' 'RHD' nan]

Unique value of:>>> Lot_Extent (129)
[ 65.  80.  68.  60.  84.  85.  75.  nan  51.  50.  70.  91.  72.  66.
 101.  57.  44. 110.  98.  47. 108. 112.  74. 115.  61.  48.  33.  52.
 100.  24.  89.  63.  76.  81.  95.  69.  21.  32.  78. 121. 122.  40.
 105.  73.  77.  64.  94.  34.  90.  55.  88.  82.  71. 120. 107.  92.
 134.  62.  86. 141.  97.  54.  41.  79. 174.  99.  67.  83.  43. 103.
  93.  30. 129. 140.  35.  37. 118.  87. 116. 150. 111.  49.  96.  59.
  36.  56. 102.  58.  38. 109. 130.  53. 137.  45. 106. 104.  42.  39.
 144. 114. 128. 149. 313. 168. 182. 138. 160. 152. 124. 153.  46.  26.
  25. 119.  31.  28. 117. 113. 125. 135. 136.  22. 123. 195. 155. 126.
 200. 131. 133.]

Unique value of:>>> Lot_Size (2532)
[ 8450.        9600.       11250.       ...  7367.775348  2203.135444
  6253.431852]

Unique value of:>>> Road_Type (2)
['Paved' 'Gravel']

Unique value of:>>> Property_Shape (4)
['Reg' 'IR1' 'IR2' 'IR3']

Unique value of:>>> Land_Outline (4)
['Lvl' 'Bnk' 'Low' 'HLS']

Unique value of:>>> Utility_Type (3)
['AllPub' 'NoSeWa' nan]

Unique value of:>>> Lot_Configuration (5)
['I' 'FR2P' 'C' 'CulDSac' 'FR3P']

Unique value of:>>> Property_Slope (3)
['GS' 'MS' 'SS']

Unique value of:>>> Neighborhood (25)
['CollgCr' 'Veenker' 'Crawfor' 'NoRidge' 'Mitchel' 'Somerst' 'NWAmes'
 'OldTown' 'BrkSide' 'Sawyer' 'NridgHt' 'NAmes' 'SawyerW' 'IDOTRR'
 'MeadowV' 'Edwards' 'Timber' 'Gilbert' 'StoneBr' 'ClearCr' 'NPkVill'
 'Blmngtn' 'BrDale' 'SWISU' 'Blueste']

Unique value of:>>> Condition1 (10)
['Norm' 'Feedr' 'PosN' 'Artery' 'RRAe' 'RRNn' 'RRAn' 'PosA' 'RRNe' 'NoRMD']

Unique value of:>>> Condition2 (9)
['Norm' 'Artery' 'RRNn' 'Feedr' 'PosN' 'PosA' 'RRAn' 'RRAe' 'NoRMD']

Unique value of:>>> House_Type (5)
['1Fam' '2fmCon' 'Duplex' 'TwnhsE' 'Twnhs']

Unique value of:>>> House_Design (8)
['2Story' '1Story' '1.5Fin' '1.5Unf' 'SFoyer' 'SLvl' '2.5Unf' '2.5Fin']

Unique value of:>>> Overall_Material (10)
[ 7  6  8  5  9  4 10  3  1  2]

Unique value of:>>> House_Condition (9)
[5 8 6 7 4 2 3 9 1]

Unique value of:>>> Construction_Year (118)
[2003 1976 2001 1915 2000 1993 2004 1973 1931 1939 1965 2005 1962 2006
 1960 1929 1970 1967 1958 1930 2002 1968 2007 1951 1957 1927 1920 1966
 1959 1994 1954 1953 1955 1983 1975 1997 1934 1963 1981 1964 1999 1972
 1921 1945 1982 1998 1956 1948 1910 1995 1991 2009 1950 1961 1977 1985
 1979 1885 1919 1990 1969 1935 1988 1971 1952 1936 1923 1924 1984 1926
 1940 1941 1987 1986 2008 1908 1892 1916 1932 1918 1912 1947 1925 1900
 1980 1989 1992 1949 1880 1928 1978 1922 1996 2010 1946 1913 1937 1942
 1938 1974 1893 1914 1906 1890 1898 1904 1882 1875 1911 1917 1872 1905
 1907 1896 1902 1895 1879 1901]

Unique value of:>>> Remodel_Year (61)
[2003 1976 2002 1970 2000 1995 2005 1973 1950 1965 2006 1962 2007 1960
 2001 1967 2004 2008 1997 1959 1990 1955 1983 1980 1966 1963 1987 1964
 1972 1996 1998 1989 1953 1956 1968 1981 1992 2009 1982 1961 1993 1999
 1985 1979 1977 1969 1958 1991 1971 1952 1975 2010 1984 1986 1994 1988
 1954 1957 1951 1978 1974]

Unique value of:>>> Roof_Design (6)
['Gable' 'Hip' 'Gambrel' 'Mansard' 'Flat' 'Shed']

Unique value of:>>> Roof_Quality (8)
['SS' 'WSh' 'ME' 'WS' 'M' 'TG' 'R' 'CT']

Unique value of:>>> Exterior1st (17)
['VinylSd' 'MetalSd' 'Wd Sdng' 'HdBoard' 'BrkFace' 'WdShing' 'CemntBd'
 'Plywood' 'AsbShng' 'Stucco' 'BrkComm' 'AsphShn' 'Stone' 'ImStucc'
 'CBlock' nan 'CB']

Unique value of:>>> Exterior2nd (17)
['VinylSd' 'MetalSd' 'Wd Shng' 'HdBoard' 'Plywood' 'Wd Sdng' 'CmentBd'
 'BrkFace' 'Stucco' 'AsbShng' 'Brk Cmn' 'ImStucc' 'AsphShn' 'Stone'
 'Other' 'CBlock' nan]

Unique value of:>>> Brick_Veneer_Type (5)
['BrkFace' 'None' 'Stone' 'BrkCmn' nan]

Unique value of:>>> Brick_Veneer_Area (445)
[1.960e+02 0.000e+00 1.620e+02 3.500e+02 1.860e+02 2.400e+02 2.860e+02
 3.060e+02 2.120e+02 1.800e+02 3.800e+02 2.810e+02 6.400e+02 2.000e+02
 2.460e+02 1.320e+02 6.500e+02 1.010e+02 4.120e+02 2.720e+02 4.560e+02
 1.031e+03 1.780e+02 5.730e+02 3.440e+02 2.870e+02 1.670e+02 1.115e+03
 4.000e+01 1.040e+02 5.760e+02 4.430e+02 4.680e+02 6.600e+01 2.200e+01
 2.840e+02 7.600e+01 2.030e+02 6.800e+01 1.830e+02 4.800e+01 2.800e+01
 3.360e+02 6.000e+02 7.680e+02 4.800e+02 2.200e+02 1.840e+02 1.129e+03
 1.160e+02 1.350e+02 2.660e+02 8.500e+01 3.090e+02 1.360e+02 2.880e+02
 7.000e+01 3.200e+02 5.000e+01 1.200e+02 4.360e+02 2.520e+02 8.400e+01
 6.640e+02 2.260e+02 3.000e+02 6.530e+02 1.120e+02 4.910e+02 2.680e+02
 7.480e+02 9.800e+01 2.750e+02 1.380e+02 2.050e+02 2.620e+02 1.280e+02
 2.600e+02 1.530e+02 6.400e+01 3.120e+02 1.600e+01 9.220e+02 1.420e+02
 2.900e+02 1.270e+02 5.060e+02 2.970e+02       nan 6.040e+02 2.540e+02
 3.600e+01 1.020e+02 4.720e+02 4.810e+02 1.080e+02 3.020e+02 1.720e+02
 3.990e+02 2.700e+02 4.600e+01 2.100e+02 1.740e+02 3.480e+02 3.150e+02
 2.990e+02 3.400e+02 1.660e+02 7.200e+01 3.100e+01 3.400e+01 2.380e+02
 1.600e+03 3.650e+02 5.600e+01 1.500e+02 2.780e+02 2.560e+02 2.250e+02
 3.700e+02 3.880e+02 1.750e+02 2.960e+02 1.460e+02 1.130e+02 1.760e+02
 6.160e+02 3.000e+01 1.060e+02 8.700e+02 3.620e+02 5.300e+02 5.000e+02
 5.100e+02 2.470e+02 3.050e+02 2.550e+02 1.250e+02 1.000e+02 4.320e+02
 1.260e+02 4.730e+02 7.400e+01 1.450e+02 2.320e+02 3.760e+02 4.200e+01
 1.610e+02 1.100e+02 1.800e+01 2.240e+02 2.480e+02 8.000e+01 3.040e+02
 2.150e+02 7.720e+02 4.350e+02 3.780e+02 5.620e+02 1.680e+02 8.900e+01
 2.850e+02 3.600e+02 9.400e+01 3.330e+02 9.210e+02 7.620e+02 5.940e+02
 2.190e+02 1.880e+02 4.790e+02 5.840e+02 1.820e+02 2.500e+02 2.920e+02
 2.450e+02 2.070e+02 8.200e+01 9.700e+01 3.350e+02 2.080e+02 4.200e+02
 1.700e+02 4.590e+02 2.800e+02 9.900e+01 1.920e+02 2.040e+02 2.330e+02
 1.560e+02 4.520e+02 5.130e+02 2.610e+02 1.640e+02 2.590e+02 2.090e+02
 2.630e+02 2.160e+02 3.510e+02 6.600e+02 3.810e+02 5.400e+01 5.280e+02
 2.580e+02 4.640e+02 5.700e+01 1.470e+02 1.170e+03 2.930e+02 6.300e+02
 4.660e+02 1.090e+02 4.100e+01 1.600e+02 2.890e+02 6.510e+02 1.690e+02
 9.500e+01 4.420e+02 2.020e+02 3.380e+02 8.940e+02 3.280e+02 6.730e+02
 6.030e+02 1.000e+00 3.750e+02 9.000e+01 3.800e+01 1.570e+02 1.100e+01
 1.400e+02 1.300e+02 1.480e+02 8.600e+02 4.240e+02 1.047e+03 2.430e+02
 8.160e+02 3.870e+02 2.230e+02 1.580e+02 1.370e+02 1.150e+02 1.890e+02
 2.740e+02 1.170e+02 6.000e+01 1.220e+02 9.200e+01 4.150e+02 7.600e+02
 2.700e+01 7.500e+01 3.610e+02 1.050e+02 3.420e+02 2.980e+02 5.410e+02
 2.360e+02 1.440e+02 4.230e+02 4.400e+01 1.510e+02 9.750e+02 4.500e+02
 2.300e+02 5.710e+02 2.400e+01 5.300e+01 2.060e+02 1.400e+01 3.240e+02
 2.950e+02 3.960e+02 6.700e+01 1.540e+02 4.250e+02 4.500e+01 1.378e+03
 3.370e+02 1.490e+02 1.430e+02 5.100e+01 1.710e+02 2.340e+02 6.300e+01
 7.660e+02 3.200e+01 8.100e+01 1.630e+02 5.540e+02 2.180e+02 6.320e+02
 1.140e+02 5.670e+02 3.590e+02 4.510e+02 6.210e+02 7.880e+02 8.600e+01
 7.960e+02 3.910e+02 2.280e+02 8.800e+01 1.650e+02 4.280e+02 4.100e+02
 5.640e+02 3.680e+02 3.180e+02 5.790e+02 6.500e+01 7.050e+02 4.080e+02
 2.440e+02 1.230e+02 3.660e+02 7.310e+02 4.480e+02 2.940e+02 3.100e+02
 2.370e+02 4.260e+02 9.600e+01 4.380e+02 1.940e+02 1.190e+02 2.000e+01
 5.040e+02 4.920e+02 6.150e+02 1.095e+03 1.159e+03 2.650e+02 9.100e+01
 7.710e+02 4.700e+01 1.770e+02 3.710e+02 4.300e+02 4.400e+02 2.290e+02
 7.260e+02 4.180e+02 7.240e+02 3.830e+02 7.300e+02 4.700e+02 3.080e+02
 6.340e+02 3.720e+02 1.980e+02 1.210e+02 2.640e+02 1.410e+02 2.830e+02
 5.090e+02 2.170e+02 3.000e+00 6.570e+02 1.240e+02 4.440e+02 2.300e+01
 2.420e+02 3.640e+02 3.520e+02 4.060e+02 4.020e+02 4.220e+02 3.560e+02
 6.800e+02 1.110e+03 2.210e+02 7.140e+02 6.470e+02 1.290e+03 4.950e+02
 5.680e+02 1.790e+02 1.050e+03 1.870e+02 5.200e+01 2.760e+02 3.900e+01
 1.900e+02 2.510e+02 2.270e+02 1.340e+02 2.220e+02 5.800e+01 6.680e+02
 6.740e+02 1.970e+02 7.100e+02 9.450e+02 5.490e+02 2.530e+02 4.000e+02
 9.700e+02 5.020e+02 3.940e+02 2.350e+02 5.150e+02 5.260e+02 7.540e+02
 3.530e+02 5.250e+02 8.700e+01 2.910e+02 6.900e+01 2.790e+02 3.230e+02
 2.140e+02 5.190e+02 1.224e+03 6.520e+02 8.860e+02 9.020e+02 4.340e+02
 6.620e+02 7.340e+02 5.500e+02 5.140e+02 3.850e+02 5.180e+02 5.720e+02
 3.220e+02 8.770e+02 3.970e+02 7.380e+02 5.010e+02 1.180e+02 6.920e+02
 3.320e+02 5.220e+02 3.790e+02 5.320e+02 6.200e+01 1.990e+02 3.550e+02
 4.050e+02 3.270e+02 2.570e+02 3.820e+02]

Unique value of:>>> Exterior_Material (4)
['Gd' 'TA' 'Ex' 'Fa']

Unique value of:>>> Exterior_Condition (5)
['TA' 'Gd' 'Fa' 'Po' 'Ex']

Unique value of:>>> Foundation_Type (6)
['PC' 'CB' 'BT' 'W' 'SL' 'S']

Unique value of:>>> Basement_Height (5)
['Gd' 'TA' 'Ex' nan 'Fa']

Unique value of:>>> Basement_Condition (5)
['TA' 'Gd' nan 'Fa' 'Po']

Unique value of:>>> Exposure_Level (5)
['No' 'Gd' 'Mn' 'Av' nan]

Unique value of:>>> BsmtFinType1 (7)
['GLQ' 'ALQ' 'Unf' 'Rec' 'BLQ' nan 'LwQ']

Unique value of:>>> BsmtFinSF1 (991)
[7.060e+02 9.780e+02 4.860e+02 2.160e+02 6.550e+02 7.320e+02 1.369e+03
 8.590e+02 0.000e+00 8.510e+02 9.060e+02 9.980e+02 7.370e+02 7.330e+02
 5.780e+02 6.460e+02 5.040e+02 8.400e+02 1.880e+02 2.340e+02 1.218e+03
 1.277e+03 1.018e+03 1.153e+03 1.213e+03 7.310e+02 6.430e+02 9.670e+02
 7.470e+02 2.800e+02 1.790e+02 4.560e+02 1.351e+03 2.400e+01 7.630e+02
 1.820e+02 1.040e+02 1.810e+03 3.840e+02 4.900e+02 6.490e+02 6.320e+02
 9.410e+02 7.390e+02 9.120e+02 1.013e+03 6.030e+02 1.880e+03 5.650e+02
 3.200e+02 4.620e+02 2.280e+02 3.360e+02 4.480e+02 1.201e+03 3.300e+01
 5.880e+02 6.000e+02 7.130e+02 1.046e+03 6.480e+02 3.100e+02 1.162e+03
 5.200e+02 1.080e+02 5.690e+02 1.200e+03 2.240e+02 7.050e+02 4.440e+02
 2.500e+02 9.840e+02 3.500e+01 7.740e+02 4.190e+02 1.700e+02 1.470e+03
 9.380e+02 5.700e+02 3.000e+02 1.200e+02 1.160e+02 5.120e+02 5.670e+02
 4.450e+02 6.950e+02 4.050e+02 1.005e+03 6.680e+02 8.210e+02 4.320e+02
 1.300e+03 5.070e+02 6.790e+02 1.332e+03 2.090e+02 6.800e+02 7.160e+02
 1.400e+03 4.160e+02 4.290e+02 2.220e+02 5.700e+01 6.600e+02 1.016e+03
 3.700e+02 3.510e+02 3.790e+02 1.288e+03 3.600e+02 6.390e+02 4.950e+02
 2.880e+02 1.398e+03 4.770e+02 8.310e+02 1.904e+03 4.360e+02 3.520e+02
 6.110e+02 1.086e+03 2.970e+02 6.260e+02 5.600e+02 3.900e+02 5.660e+02
 1.126e+03 1.036e+03 1.088e+03 6.410e+02 6.170e+02 6.620e+02 3.120e+02
 1.065e+03 7.870e+02 4.680e+02 3.600e+01 8.220e+02 3.780e+02 9.460e+02
 3.410e+02 1.600e+01 5.500e+02 5.240e+02 5.600e+01 3.210e+02 8.420e+02
 6.890e+02 6.250e+02 3.580e+02 4.020e+02 9.400e+01 1.078e+03 3.290e+02
 9.290e+02 6.970e+02 1.573e+03 2.700e+02 9.220e+02 5.030e+02 1.334e+03
 3.610e+02 6.720e+02 5.060e+02 7.140e+02 4.030e+02 7.510e+02 2.260e+02
 6.200e+02 5.460e+02 3.920e+02 4.210e+02 9.050e+02 9.040e+02 4.300e+02
 6.140e+02 4.500e+02 2.100e+02 2.920e+02 7.950e+02 1.285e+03 8.190e+02
 4.200e+02 8.410e+02 2.810e+02 8.940e+02 1.464e+03 7.000e+02 2.620e+02
 1.274e+03 5.180e+02 1.236e+03 4.250e+02 6.920e+02 9.870e+02 9.700e+02
 2.800e+01 2.560e+02 1.619e+03 4.000e+01 8.460e+02 1.124e+03 7.200e+02
 8.280e+02 1.249e+03 8.100e+02 2.130e+02 5.850e+02 1.290e+02 4.980e+02
 1.270e+03 5.730e+02 1.410e+03 1.082e+03 2.360e+02 3.880e+02 3.340e+02
 8.740e+02 9.560e+02 7.730e+02 3.990e+02 1.620e+02 7.120e+02 6.090e+02
 3.710e+02 5.400e+02 7.200e+01 6.230e+02 4.280e+02 3.500e+02 2.980e+02
 1.445e+03 2.180e+02 9.850e+02 6.310e+02 1.280e+03 2.410e+02 6.900e+02
 2.660e+02 7.770e+02 8.120e+02 7.860e+02 1.116e+03 7.890e+02 1.056e+03
 5.000e+01 1.128e+03 7.750e+02 1.309e+03 1.246e+03 9.860e+02 6.160e+02
 1.518e+03 6.640e+02 3.870e+02 4.710e+02 3.850e+02 3.650e+02 1.767e+03
 1.330e+02 6.420e+02 2.470e+02 3.310e+02 7.420e+02 1.606e+03 9.160e+02
 1.850e+02 5.440e+02 5.530e+02 3.260e+02 7.780e+02 3.860e+02 4.260e+02
 3.680e+02 4.590e+02 1.350e+03 1.196e+03 6.300e+02 9.940e+02 1.680e+02
 1.261e+03 1.567e+03 2.990e+02 8.970e+02 6.070e+02 8.360e+02 5.150e+02
 3.740e+02 1.231e+03 1.110e+02 3.560e+02 4.000e+02 6.980e+02 1.247e+03
 2.570e+02 3.800e+02 2.700e+01 1.410e+02 9.910e+02 6.500e+02 5.210e+02
 1.436e+03 2.260e+03 7.190e+02 3.770e+02 1.330e+03 3.480e+02 1.219e+03
 7.830e+02 9.690e+02 6.730e+02 1.358e+03 1.260e+03 1.440e+02 5.840e+02
 5.540e+02 1.002e+03 6.190e+02 1.800e+02 5.590e+02 3.080e+02 8.660e+02
 8.950e+02 6.370e+02 6.040e+02 1.302e+03 1.071e+03 2.900e+02 7.280e+02
 2.000e+00 1.441e+03 9.430e+02 2.310e+02 4.140e+02 3.490e+02 4.420e+02
 3.280e+02 5.940e+02 8.160e+02 1.460e+03 1.324e+03 1.338e+03 6.850e+02
 1.422e+03 1.283e+03 8.100e+01 4.540e+02 9.030e+02 6.050e+02 9.900e+02
 2.060e+02 1.500e+02 4.570e+02 4.800e+01 8.710e+02 4.100e+01 6.740e+02
 6.240e+02 4.800e+02 1.154e+03 7.380e+02 4.930e+02 1.121e+03 2.820e+02
 5.000e+02 1.310e+02 1.696e+03 8.060e+02 1.361e+03 9.200e+02 1.721e+03
 1.870e+02 1.138e+03 9.880e+02 1.930e+02 5.510e+02 7.670e+02 1.186e+03
 8.920e+02 3.110e+02 8.270e+02 5.430e+02 1.003e+03 1.059e+03 2.390e+02
 9.450e+02 2.000e+01 1.455e+03 9.650e+02 9.800e+02 8.630e+02 5.330e+02
 1.084e+03 1.173e+03 5.230e+02 1.148e+03 1.910e+02 1.234e+03 3.750e+02
 8.080e+02 7.240e+02 1.520e+02 1.180e+03 2.520e+02 8.320e+02 5.750e+02
 9.190e+02 4.390e+02 3.810e+02 4.380e+02 5.490e+02 6.120e+02 1.163e+03
 4.370e+02 3.940e+02 1.416e+03 4.220e+02 7.620e+02 9.750e+02 1.097e+03
 2.510e+02 6.860e+02 6.560e+02 5.680e+02 5.390e+02 8.620e+02 1.970e+02
 5.160e+02 6.630e+02 6.080e+02 1.636e+03 7.840e+02 2.490e+02 1.040e+03
 4.830e+02 1.960e+02 5.720e+02 3.380e+02 3.300e+02 1.560e+02 1.390e+03
 5.130e+02 4.600e+02 6.590e+02 3.640e+02 5.640e+02 3.060e+02 5.050e+02
 9.320e+02 7.500e+02 6.400e+01 6.330e+02 1.170e+03 8.990e+02 9.020e+02
 1.238e+03 5.280e+02 1.024e+03 1.064e+03 2.850e+02 2.188e+03 4.650e+02
 3.220e+02 8.600e+02 5.990e+02 3.540e+02 6.300e+01 2.230e+02 3.010e+02
 4.430e+02 4.890e+02 2.840e+02 2.940e+02 8.140e+02 1.650e+02 5.520e+02
 8.330e+02 4.640e+02 9.360e+02 7.720e+02 1.440e+03 7.480e+02 9.820e+02
 3.980e+02 5.620e+02 4.840e+02 4.170e+02 6.990e+02 6.960e+02 8.960e+02
 5.560e+02 1.106e+03 6.510e+02 8.670e+02 8.540e+02 1.646e+03 1.074e+03
 5.360e+02 1.172e+03 9.150e+02 5.950e+02 1.237e+03 2.730e+02 6.840e+02
 3.240e+02 1.165e+03 1.380e+02 1.513e+03 3.170e+02 1.012e+03 1.022e+03
 5.090e+02 9.000e+02 1.085e+03 1.104e+03 2.400e+02 3.830e+02 6.440e+02
 3.970e+02 7.400e+02 8.370e+02 2.200e+02 5.860e+02 5.350e+02 4.100e+02
 7.500e+01 8.240e+02 5.920e+02 1.039e+03 5.100e+02 4.230e+02 6.610e+02
 2.480e+02 7.040e+02 4.120e+02 1.032e+03 2.190e+02 7.080e+02 4.150e+02
 1.004e+03 3.530e+02 7.020e+02 3.690e+02 6.220e+02 2.120e+02 6.450e+02
 8.520e+02 1.150e+03 1.258e+03 2.750e+02 1.760e+02 2.960e+02 5.380e+02
 1.157e+03 4.920e+02 1.198e+03 1.387e+03 5.220e+02 6.580e+02 1.216e+03
 1.480e+03 2.096e+03 1.159e+03 4.400e+02 1.456e+03 8.830e+02 5.470e+02
 7.880e+02 4.850e+02 3.400e+02 1.220e+03 4.270e+02 3.440e+02 7.560e+02
 1.540e+03 6.660e+02 8.030e+02 1.000e+03 8.850e+02 1.386e+03 3.190e+02
 5.340e+02 1.250e+02 1.314e+03 6.020e+02 1.920e+02 5.930e+02 8.040e+02
 1.053e+03 5.320e+02 1.158e+03 1.014e+03 1.940e+02 1.670e+02 7.760e+02
 5.644e+03 6.940e+02 1.572e+03 7.460e+02 1.406e+03 9.250e+02 4.820e+02
 1.890e+02 7.650e+02 8.000e+01 1.443e+03 2.590e+02 7.350e+02 7.340e+02
 1.447e+03 5.480e+02 3.150e+02 1.282e+03 4.080e+02 3.090e+02 2.030e+02
 8.650e+02 2.040e+02 7.900e+02 1.320e+03 7.690e+02 1.070e+03 2.640e+02
 7.590e+02 1.373e+03 9.760e+02 7.810e+02 2.500e+01 1.110e+03 4.040e+02
 5.800e+02 6.780e+02 9.580e+02 1.336e+03 1.079e+03 4.900e+01 9.230e+02
 7.910e+02 2.630e+02 9.350e+02 1.051e+03 5.140e+02 1.100e+02 1.414e+03
 1.260e+02 1.129e+03 1.298e+03 3.760e+02 4.660e+02 2.440e+02 1.137e+03
 6.870e+02 1.010e+03 1.500e+03 6.700e+02 9.440e+02 1.188e+03 8.560e+02
 3.390e+02 4.810e+02 7.170e+02 5.790e+02 2.740e+02 7.800e+02 2.830e+02
 4.740e+02 4.520e+02 2.760e+02 9.600e+02 7.660e+02 1.026e+03 7.300e+01
 7.360e+02 1.319e+03 2.670e+02 1.092e+03 9.640e+02 9.540e+02 1.346e+03
 1.433e+03 8.700e+02 1.980e+02 1.682e+03 2.380e+02 3.430e+02 7.600e+01
 6.150e+02 7.800e+01 4.200e+01 4.690e+02 2.070e+02 4.580e+02 4.760e+02
 1.341e+03 8.440e+02 8.470e+02 8.500e+02 1.965e+03 7.410e+02 3.630e+02
 2.250e+02 1.333e+03 8.880e+02 6.360e+02 7.260e+02 2.540e+02 4.350e+02
 3.890e+02 2.790e+02 1.360e+03 1.232e+03 2.288e+03 1.531e+03 1.230e+03
 1.015e+03 1.037e+03 1.142e+03 1.262e+03 1.972e+03 8.810e+02 8.760e+02
 2.146e+03 1.557e+03 8.000e+02 6.520e+02 4.940e+02 6.830e+02 9.130e+02
 1.294e+03 2.158e+03 6.820e+02 1.430e+03 7.710e+02 5.400e+01 5.200e+01
 6.800e+01 8.640e+02 1.400e+02 1.733e+03 6.010e+02 9.620e+02 1.252e+03
 1.210e+02 9.550e+02 1.000e+02 1.312e+03 1.720e+02 1.550e+02 9.310e+02
 8.720e+02 7.450e+02 6.210e+02 4.330e+02 8.260e+02 1.340e+02 1.690e+02
 7.490e+02 1.152e+03 5.270e+02 3.420e+02 1.730e+02 7.000e+01 1.094e+03
 8.200e+02 1.021e+03 1.359e+03 7.550e+02 9.500e+02 6.060e+02 1.259e+03
 7.100e+02 1.111e+03 1.478e+03 3.320e+02 7.930e+02 2.460e+02 1.540e+02
 6.500e+01 1.476e+03 5.500e+01 1.758e+03 1.115e+03 1.640e+03 1.140e+02
 7.180e+02 4.960e+02 1.337e+03 1.034e+03 9.830e+02 1.206e+03 8.900e+02
 1.023e+03 1.190e+02 2.860e+02 1.728e+03 1.375e+03 1.420e+03 2.257e+03
 1.149e+03 1.075e+03 3.720e+02 1.204e+03 1.073e+03 1.087e+03 1.660e+03
 1.096e+03 7.290e+02 3.620e+02 5.370e+02 4.720e+02 5.300e+01 7.640e+02
 1.900e+02 1.027e+03 1.141e+03 6.810e+02 8.130e+02 1.280e+02 1.044e+03
 2.600e+02 5.830e+02 3.200e+01 5.310e+02 1.480e+02 7.440e+02 9.600e+01
 5.900e+02 2.000e+02 4.060e+02 1.750e+02 2.010e+02       nan 7.580e+02
 2.210e+02 6.340e+02 1.035e+03 7.790e+02 1.271e+03 3.550e+02 2.085e+03
 7.700e+02 7.220e+02 1.308e+03 6.880e+02 8.800e+01 1.194e+03 1.538e+03
 1.593e+03 1.033e+03 3.660e+02 1.474e+03 1.383e+03 8.930e+02 1.029e+03
 1.223e+03 1.011e+03 1.571e+03 3.180e+02 5.010e+02 7.850e+02 6.380e+02
 6.470e+02 8.380e+02 1.860e+02 9.260e+02 1.101e+03 1.047e+03 7.970e+02
 1.558e+03 1.328e+03 3.140e+02 9.300e+02 7.250e+02 1.151e+03 1.304e+03
 1.812e+03 1.684e+03 6.690e+02 1.178e+03 1.030e+03 8.480e+02 9.180e+02
 5.740e+02 1.181e+03 1.048e+03 3.350e+02 1.225e+03 7.270e+02 9.680e+02
 6.000e+01 9.370e+02 9.010e+02 1.732e+03 1.632e+03 9.730e+02 9.100e+02
 3.460e+02 7.920e+02 6.540e+02 1.300e+02 8.730e+02 9.080e+02 4.410e+02
 8.500e+01 2.420e+02 9.520e+02 1.098e+03 7.820e+02 1.220e+02 3.160e+02
 2.580e+02 5.870e+02 4.910e+02 4.530e+02 5.570e+02 1.080e+03 4.970e+02
 5.100e+01 5.020e+02 6.710e+02 1.412e+03 7.090e+02 1.320e+02 4.010e+03
 4.670e+02 7.700e+01 1.130e+02 5.770e+02 4.340e+02 1.001e+03 1.392e+03
 1.239e+03 9.240e+02 9.490e+02 2.150e+02 1.329e+03 1.112e+03 7.960e+02
 8.110e+02 1.090e+03 5.960e+02 1.127e+03 2.050e+02 1.191e+03 9.510e+02
 3.820e+02 3.730e+02 1.505e+03 1.290e+03 8.800e+02 1.038e+03 1.182e+03
 1.562e+03 1.836e+03 2.780e+02 1.810e+02 1.118e+03 7.600e+02 7.990e+02
 9.960e+02 9.390e+02 9.140e+02 2.710e+02 4.880e+02 7.010e+02 4.550e+02
 8.090e+02 9.530e+02 2.080e+02 1.430e+02 5.760e+02 3.470e+02 7.940e+02
 2.300e+02 2.610e+02 3.930e+02 1.576e+03 1.122e+03 8.530e+02 4.750e+02
 6.910e+02 4.240e+02 3.050e+02 5.260e+02 1.564e+03 9.090e+02 1.136e+03
 1.243e+03 1.490e+02 1.224e+03 3.370e+02]

Unique value of:>>> BsmtFinType2 (7)
['Unf' 'BLQ' nan 'ALQ' 'Rec' 'LwQ' 'GLQ']

Unique value of:>>> BsmtFinSF2 (273)
[   0.   32.  668.  486.   93.  491.  506.  712.  362.   41.  169.  869.
  150.  670.   28. 1080.  181.  768.  215.  374.  208.  441.  184.  279.
  306.  180.  580.  690.  692.  228.  125. 1063.  620.  175.  820. 1474.
  264.  479.  147.  232.  380.  544.  294.  258.  121.  391.  531.  344.
  539.  713.  210.  311. 1120.  165.  532.   96.  495.  174. 1127.  139.
  202.  645.  123.  551.  219.  606.  612.  480.  182.  132.  336.  468.
  287.   35.  499.  723.  119.   40.  117.  239.   80.  472.   64. 1057.
  127.  630.  128.  377.  764.  345. 1085.  435.  823.  500.  290.  324.
  634.  411.  841. 1061.  466.  396.  354.  149.  193.  273.  465.  400.
  682.  557.  230.  106.  791.  240.  547.  469.  177.  108.  600.  492.
  211.  168. 1031.  438.  375.  144.   81.  906.  608.  276.  661.   68.
  173.  972.  105.  420.  546.  334.  352.  872.  110.  627.  163. 1029.
   78.  859.  981.   42.   46.  162.  350.  263. 1073.   12.  159.  474.
  453.  684.  387.  688.  252.  590.  284.  622.  113. 1526.  360.  774.
  364.  596.  884.   92.  216.  136.  201.  512.  247.  483.  750.   60.
  102.   95.   63.  262.  393.  286.  450.   72.  243.  694.  875.  507.
  419.  250.  116.  624.   76.  270.  288.  186.  449.   48.  613.  852.
  555.  799.  811.  842.  382.  456.  308.   52.  196.  488.  319.   nan
  956.  120.  679.  604.  153.  619.    6.  351. 1037.  829.   38.  206.
  167.  543.  259.  404.  138.  955.  691.   66.  154.  442.  448.  227.
  398.  722.  761.  529.  522.  873.  891.  755.  321.  915.  417.  432.
  831.  278. 1020.  530.  904.  156. 1393. 1039.  497.  402.  748.  281.
  912.  373.  982.  826.  850. 1164. 1083.  337.  297.]

Unique value of:>>> BsmtUnfSF (1136)
[ 150.  284.  434. ...  129.   45. 1503.]

Unique value of:>>> Total_Basement_Area (1059)
[ 856. 1262.  920. ...  498.  432. 1381.]

Unique value of:>>> Heating_Type (6)
['GasA' 'GasW' 'Grav' 'Wall' 'OthW' 'Floor']

Unique value of:>>> Heating_Quality (5)
['Ex' 'Gd' 'TA' 'Fa' 'Po']

Unique value of:>>> Air_Conditioning (2)
['Y' 'N']

Unique value of:>>> Electrical_System (6)
['SBrkr' 'FuseF' 'FuseA' 'FuseP' 'Mix' nan]

Unique value of:>>> First_Floor_Area (1083)
[ 856 1262  920 ... 1778 1650 1960]

Unique value of:>>> Second_Floor_Area (635)
[ 854    0  866  756 1053  566  983  752 1142 1218  668 1320  631  716
  676  860 1519  530  808  977 1330  833  765  462  213  548  960  670
 1116  876  612 1031  881  790  755  592  939  520  639  656 1414  884
  729 1523  728  351  688  941 1032  848  836  475  739 1151  448  896
  524 1194  956 1070 1096  467  547  551  880  703  901  720  316 1518
  704 1178  754  601 1360  929  445  564  882  920  518  817 1257  741
  672 1306  504 1304 1100  730  689  591  888 1020  828  700  842 1286
  864  829 1092  709  844 1106  596  807  625  649  698  840  780  568
  795  648  975  702 1242 1818 1121  371  804  325  809 1200  871 1274
 1347 1332 1177 1080  695  167  915  576  605  862  495  403  838  517
 1427  784  711  468 1081  886  793  665  858  874  526  590  406 1157
  299  936  438 1098  766 1101 1028 1017 1254  378 1160  682  110  600
  678  834  384  512  930  868  224 1103  560  811  878  574  910  620
  687  546  902 1000  846 1067  914  660 1538 1015 1237  611  707  527
 1288  832  806 1182 1040  439  717  511 1129 1370  636  533  745  584
  812  684  595  988  800  677  573 1066  778  661 1440  872  788  843
  713  567  651  762  482  738  586  679  644  900  887 1872 1281  472
 1312  319  978 1093  473  664 1540 1276  441  348 1060  714  744 1203
  783 1097  734  767 1589  742  686 1128 1111 1174  787 1072 1088 1063
  545  966  623  432  581  540  769 1051  761  779  514  455 1426  785
  521  252  813 1120 1037 1169 1001 1215  928 1140 1243  571 1196 1038
  561  979  701  332  368  883 1336 1141  634  912  798  985  826  831
  750  456  602  855  336  408  980  998 1168 1208  797  850  898 1054
  895  954  772 1230  727  454  370  628  304  582 1122 1134  885  640
  580 1112  653  220  240 1362  534  539  650  918  933  712 1796  971
 1175  743  523 1216 2065  272  685  776  630  984  875  913  464 1039
 1259  940  892  725  924  764  925 1479  192  589  992  903  430  748
  587  994  950 1323  732 1357  557 1296  390 1185  873 1611  457  796
  908  550  989  932  358 1392  349  691 1349  768  208  622  857  556
 1044  708  626  904  510 1104  830  981  870  694 1152  563  823  604
  715  532  537  505  424  606  185  498  492  608 1074  662  499  180
  942  558  614  328 1788 1075  380  615  645  663 1275  816  839 1325
 1012 1295  683 1126 1089 1221  967  841 1209  897  786 1629  782 1369
  972 1315  726  322  760  629  496  690  646  917  624  320  588  425
  747 1114 1619  718  815  926  444  436 1240  516 1420 1158 1162 1139
 1285 1061 1250  919  861  794  825  893 1319  959  792 1345  453  412
  182  501  375  680  658  552  396  308  973  363  594  554  428  536
  486 1721 1099  735  899 1198  343  673  442  890  943  330  420  770
 1342 1377  845 1402 1036  570 1238  923  757 1048 1131 1407 1171 1277
  995  528  863 1232  976 1008 1309  228  500  544 1778  616  494  642
  659  671  144  525  423 1164  356  245 1042  477 1005 1087  638  400
  376  916  927  869  753  450 1133  674  125  531  585  775  851  957
 1340  955  990 1384 1862 1371 1405 1358  465  466 1335  814  488 1321
 1029 1368 1567 1189 1234 1248  821 1007  476  502  867  297  810  434
  583  341 1836  541 1246 1124 1045  827 1150  312  218  493  736  818
  610  549  697  360 1004]

Unique value of:>>> LowQualFinSF (36)
[   0  360  513  234  528  572  144  392  371  390  420  473  156  515
   80   53  232  481  120  514  397  479  205  384  362 1064  431  436
  259  312  108  697  512  114  140  450]

Unique value of:>>> Grade_Living_Area (1292)
[1710 1262 1786 ... 2315  641 1778]

Unique value of:>>> Underground_Full_Bathroom (5)
[ 1.  0.  2.  3. nan]

Unique value of:>>> Underground_Half_Bathroom (4)
[ 0.  1.  2. nan]

Unique value of:>>> Full_Bathroom_Above_Grade (5)
[2 1 3 0 4]

Unique value of:>>> Half_Bathroom_Above_Grade (3)
[1 0 2]

Unique value of:>>> Bedroom_Above_Grade (8)
[3 4 1 2 0 5 6 8]

Unique value of:>>> Kitchen_Above_Grade (4)
[1 2 3 0]

Unique value of:>>> Kitchen_Quality (5)
['Gd' 'TA' 'Ex' 'Fa' nan]

Unique value of:>>> Rooms_Above_Grade (14)
[ 8  6  7  9  5 11  4 10 12  3  2 14 13 15]

Unique value of:>>> Functional_Rate (11)
['TF' 'MD1' 'MajD1' 'MD2' 'MD' 'MajD2' 'SD' 'MS' 'Mod' 'Sev' nan]

Unique value of:>>> Fireplaces (5)
[0 1 2 3 4]

Unique value of:>>> Garage (8)
['Attchd' 'Detchd' 'BuiltIn' 'CarPort' nan 'Basment' '2TFes' '2Types']

Unique value of:>>> Garage_Built_Year (104)
[2003. 1976. 2001. 1998. 2000. 1993. 2004. 1973. 1931. 1939. 1965. 2005.
 1962. 2006. 1960. 1991. 1970. 1967. 1958. 1930. 2002. 1968. 2007. 2008.
 1957. 1920. 1966. 1959. 1995. 1954. 1953.   nan 1983. 1977. 1997. 1985.
 1963. 1981. 1964. 1999. 1935. 1990. 1945. 1987. 1989. 1915. 1956. 1948.
 1974. 2009. 1950. 1961. 1921. 1900. 1979. 1951. 1969. 1936. 1975. 1971.
 1923. 1984. 1926. 1955. 1986. 1988. 1916. 1932. 1972. 1918. 1980. 1924.
 1996. 1940. 1949. 1994. 1910. 1978. 1982. 1992. 1925. 1941. 2010. 1927.
 1947. 1937. 1942. 1938. 1952. 1928. 1922. 1934. 1906. 1914. 1946. 1908.
 1929. 1933. 1917. 1896. 1895. 2207. 1943. 1919.]

Unique value of:>>> Garage_Finish_Year (4)
['RFn' 'Unf' 'Fin' nan]

Unique value of:>>> Garage_Size (7)
[ 2.  3.  1.  0.  4.  5. nan]

Unique value of:>>> Garage_Area (1919)
[1085.793744   196.3163044  218.0684028 ...  518.         714.
  682.       ]

Unique value of:>>> Garage_Quality (6)
['TA' 'Fa' 'Gd' nan 'Ex' 'Po']

Unique value of:>>> Garage_Condition (6)
['TA' 'Fa' nan 'Gd' 'Po' 'Ex']

Unique value of:>>> Pavedd_Drive (3)
['Y' 'N' 'P']

Unique value of:>>> W_Deck_Area (1722)
[163.7880797  198.9000744   26.12753268 ... 197.         530.
 474.        ]

Unique value of:>>> Open_Lobby_Area (1662)
[ 69.59611493  74.71603269  32.08526783 ... 170.         274.
 225.        ]

Unique value of:>>> Enclosed_Lobby_Area (1590)
[ 20.33793445  15.03939163 -46.23219756 ... 429.         132.
  23.        ]

Unique value of:>>> Three_Season_Lobby_Area (31)
[  0 320 407 130 180 168 140 508 238 245 196 144 182 162  23 216  96 153
 290 304 224 255 225 360 150 174 120 219 176  86 323]

Unique value of:>>> Screen_Lobby_Area (121)
[  0 176 198 291 252  99 184 168 130 142 192 410 224 266 170 154 153 144
 128 259 160 271 234 374 185 182  90 396 140 276 180 161 145 200 122  95
 120  60 126 189 260 147 385 287 156 100 216 210 197 204 225 152 175 312
 222 265 322 190 233  63  53 143 273 288 263  80 163 116 480 178 440 155
 220 119 165  40 256 240 148 166 108 490 196 121  92 342 255 111 112 231
 110 117 195 115 141 208  94 164  64 576 227 221 171 135 174 217 201 109
 150  84 228 138  88 280 123 264 270 162 348 113 104]

Unique value of:>>> Pool_Area (14)
[  0 512 648 576 555 480 519 738 144 368 444 228 561 800]

Unique value of:>>> Miscellaneous_Value (38)
[    0   700   350   500   400   480   450 15500  1200   800  2000   600
  3500  1300    54   620   560  1400  8300  1150  2500 12500  1500   300
    80   490   650   900   750  6500  1000  4500  3000 17000  1512   455
   460   420]

Unique value of:>>> Month_Sold (12)
[ 2  5  9 12 10  8 11  4  1  7  3  6]

Unique value of:>>> Year_Sold (5)
[2008 2007 2006 2009 2010]

Unique value of:>>> Sale_Type (10)
['WD' 'New' 'COD' 'ConLD' 'ConLI' 'CWD' 'ConLw' 'Con' 'Oth' nan]

Unique value of:>>> Sale_Condition (8)
['Normal' 'Abnorml' 'Partial' 'AdjLand' 'Alloca' 'Family' 'NoRMDal'
 'AbnoRMDl']

# Describe the target 
train["Sale_Price"].describe()
count      1459.000000
mean     180944.102810
std       79464.918335
min       34900.000000
25%      129950.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
Name: Sale_Price, dtype: float64
# Plot the distplot of target
plt.figure(figsize=(10,8))
bar = sns.distplot(train["Sale_Price"])
bar.legend(["Skewness: {:.2f}".format(train['Sale_Price'].skew())])
C:\ProgramData\Anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
  warnings.warn(msg, FutureWarning)
<matplotlib.legend.Legend at 0x1d173dadb80>

# correlation heatmap
plt.figure(figsize=(25,25))
ax = sns.heatmap(train.corr(), cmap = "coolwarm", annot=True, linewidth=2)

# to fix the bug "first and last row cut in half of heatmap plot"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
(38.5, -0.5)

# correlation heatmap of higly correlated features with SalePrice
hig_corr = train.corr()
hig_corr_features = hig_corr.index[abs(hig_corr["Sale_Price"]) >= 0.5]
hig_corr_features
Index(['Overall_Material', 'Construction_Year', 'Remodel_Year',
       'Total_Basement_Area', 'First_Floor_Area', 'Grade_Living_Area',
       'Full_Bathroom_Above_Grade', 'Rooms_Above_Grade', 'Garage_Size',
       'Sale_Price'],
      dtype='object')
plt.figure(figsize=(10,8))
ax = sns.heatmap(train[hig_corr_features].corr(), cmap = "coolwarm", annot=True, linewidth=3)
# to fix the bug "first and last row cut in half of heatmap plot"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
(10.5, -0.5)

# Plot regplot to get the nature of highly correlated data
plt.figure(figsize=(16,9))
for i in range(len(hig_corr_features)):
    if i <= 9:
        plt.subplot(3,4,i+1)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        sns.regplot(data=train, x = hig_corr_features[i], y = 'Sale_Price')

## Handling Missing Value
missing_col = df.columns[df.isnull().any()]
missing_col
Index(['Zoning_Class', 'Lot_Extent', 'Utility_Type', 'Exterior1st',
       'Exterior2nd', 'Brick_Veneer_Type', 'Brick_Veneer_Area',
       'Basement_Height', 'Basement_Condition', 'Exposure_Level',
       'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
       'Total_Basement_Area', 'Electrical_System', 'Underground_Full_Bathroom',
       'Underground_Half_Bathroom', 'Kitchen_Quality', 'Functional_Rate',
       'Garage', 'Garage_Built_Year', 'Garage_Finish_Year', 'Garage_Size',
       'Garage_Area', 'Garage_Quality', 'Garage_Condition', 'Sale_Type'],
      dtype='object')
Handling missing value of Bsmt feature
Zoning_Col = ['Zoning_Class', 'Lot_Extent', 'Utility_Type', 'Exterior1st',
       'Exterior2nd', 'Brick_Veneer_Type', 'Brick_Veneer_Area',
       'Basement_Height', 'Basement_Condition', 'Exposure_Level']
Zoning_Col = df[Zoning_Col]
Zoning_Col
['Zoning_Class',
 'Lot_Extent',
 'Utility_Type',
 'Exterior1st',
 'Exterior2nd',
 'Brick_Veneer_Type',
 'Brick_Veneer_Area',
 'Basement_Height',
 'Basement_Condition',
 'Exposure_Level']
Zoning_Col.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 2918 entries, 1 to 2919
Data columns (total 10 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   Zoning_Class        2914 non-null   object 
 1   Lot_Extent          2432 non-null   float64
 2   Utility_Type        2916 non-null   object 
 3   Exterior1st         2917 non-null   object 
 4   Exterior2nd         2917 non-null   object 
 5   Brick_Veneer_Type   2894 non-null   object 
 6   Brick_Veneer_Area   2895 non-null   float64
 7   Basement_Height     2837 non-null   object 
 8   Basement_Condition  2836 non-null   object 
 9   Exposure_Level      2836 non-null   object 
dtypes: float64(2), object(8)
memory usage: 250.8+ KB
Zoning_Col.isnull().sum()
Zoning_Class            4
Lot_Extent            486
Utility_Type            2
Exterior1st             1
Exterior2nd             1
Brick_Veneer_Type      24
Brick_Veneer_Area      23
Basement_Height        81
Basement_Condition     82
Exposure_Level         82
dtype: int64
Zoning_Col = Zoning_Col[Zoning_Col.isnull().any(axis=1)]
Zoning_Col
Zoning_Class	Lot_Extent	Utility_Type	Exterior1st	Exterior2nd	Brick_Veneer_Type	Brick_Veneer_Area	Basement_Height	Basement_Condition	Exposure_Level
Id										
8	RLD	NaN	AllPub	HdBoard	HdBoard	Stone	240.0	Gd	TA	Mn
13	RLD	NaN	AllPub	HdBoard	Plywood	None	0.0	TA	TA	No
15	RLD	NaN	AllPub	MetalSd	MetalSd	BrkFace	212.0	TA	TA	No
17	RLD	NaN	AllPub	Wd Sdng	Wd Sdng	BrkFace	180.0	TA	TA	No
18	RLD	72.0	AllPub	MetalSd	MetalSd	None	0.0	NaN	NaN	NaN
...	...	...	...	...	...	...	...	...	...	...
2892	Commer	69.0	AllPub	Wd Sdng	Wd Sdng	None	0.0	NaN	NaN	NaN
2901	RLD	NaN	AllPub	Plywood	Plywood	None	0.0	Gd	TA	Gd
2902	RLD	NaN	AllPub	VinylSd	VinylSd	None	0.0	Gd	TA	Av
2905	NaN	125.0	AllPub	CB	VinylSd	None	0.0	NaN	NaN	NaN
2909	RLD	NaN	AllPub	Plywood	Plywood	None	0.0	TA	TA	No
579 rows × 10 columns

Zoning_Col_all_nan = Zoning_Col[(Zoning_Col.isnull() | Zoning_Col.isin([0])).all(1)]
Zoning_Col_all_nan
Zoning_Class	Lot_Extent	Utility_Type	Exterior1st	Exterior2nd	Brick_Veneer_Type	Brick_Veneer_Area	Basement_Height	Basement_Condition	Exposure_Level
Id										
Zoning_Col_all_nan.shape
(0, 10)
qual = list(df.loc[:, df.dtypes == 'object'].columns.values)
qual
['Zoning_Class',
 'Road_Type',
 'Property_Shape',
 'Land_Outline',
 'Utility_Type',
 'Lot_Configuration',
 'Property_Slope',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'House_Type',
 'House_Design',
 'Roof_Design',
 'Roof_Quality',
 'Exterior1st',
 'Exterior2nd',
 'Brick_Veneer_Type',
 'Exterior_Material',
 'Exterior_Condition',
 'Foundation_Type',
 'Basement_Height',
 'Basement_Condition',
 'Exposure_Level',
 'BsmtFinType1',
 'BsmtFinType2',
 'Heating_Type',
 'Heating_Quality',
 'Air_Conditioning',
 'Electrical_System',
 'Kitchen_Quality',
 'Functional_Rate',
 'Garage',
 'Garage_Finish_Year',
 'Garage_Quality',
 'Garage_Condition',
 'Pavedd_Drive',
 'Sale_Type',
 'Sale_Condition']
# Fillinf the mising value in bsmt features
for i in Zoning_Col:
    if i in qual:
        Zoning_Col_all_nan[i] = Zoning_Col_all_nan[i].replace(np.nan, 'NA') # replace the NAN value by 'NA'
    else:
        Zoning_Col_all_nan[i] = Zoning_Col_all_nan[i].replace(np.nan, 0) # replace the NAN value inplace of 0

Zoning_Col.update(Zoning_Col_all_nan) # update bsmt_feat df by bsmt_feat_all_nan
df.update(Zoning_Col_all_nan) # update df by bsmt_feat_all_nan

"""
>>> df = pd.DataFrame({'A': [1, 2, 3],
...                    'B': [400, 500, 600]})
>>> new_df = pd.DataFrame({'B': [4, 5, 6],
...                        'C': [7, 8, 9]})
>>> df.update(new_df)
>>> df
   A  B
0  1  4
1  2  5
2  3  6
"""
"\n>>> df = pd.DataFrame({'A': [1, 2, 3],\n...                    'B': [400, 500, 600]})\n>>> new_df = pd.DataFrame({'B': [4, 5, 6],\n...                        'C': [7, 8, 9]})\n>>> df.update(new_df)\n>>> df\n   A  B\n0  1  4\n1  2  5\n2  3  6\n"
Zoning_Col = Zoning_Col[Zoning_Col.isin([np.nan]).any(axis=1)]
Zoning_Col
Zoning_Class	Lot_Extent	Utility_Type	Exterior1st	Exterior2nd	Brick_Veneer_Type	Brick_Veneer_Area	Basement_Height	Basement_Condition	Exposure_Level
Id										
8	RLD	NaN	AllPub	HdBoard	HdBoard	Stone	240.0	Gd	TA	Mn
13	RLD	NaN	AllPub	HdBoard	Plywood	None	0.0	TA	TA	No
15	RLD	NaN	AllPub	MetalSd	MetalSd	BrkFace	212.0	TA	TA	No
17	RLD	NaN	AllPub	Wd Sdng	Wd Sdng	BrkFace	180.0	TA	TA	No
18	RLD	72.0	AllPub	MetalSd	MetalSd	None	0.0	NaN	NaN	NaN
...	...	...	...	...	...	...	...	...	...	...
2892	Commer	69.0	AllPub	Wd Sdng	Wd Sdng	None	0.0	NaN	NaN	NaN
2901	RLD	NaN	AllPub	Plywood	Plywood	None	0.0	Gd	TA	Gd
2902	RLD	NaN	AllPub	VinylSd	VinylSd	None	0.0	Gd	TA	Av
2905	NaN	125.0	AllPub	CB	VinylSd	None	0.0	NaN	NaN	NaN
2909	RLD	NaN	AllPub	Plywood	Plywood	None	0.0	TA	TA	No
579 rows × 10 columns

Zoning_Col.shape
(579, 10)
print(df['BsmtFinSF2'].max())
print(df['BsmtFinSF2'].min())
1526.0
0.0
pd.cut(range(0,1526),5) # create a bucket
[(-1.525, 305.0], (-1.525, 305.0], (-1.525, 305.0], (-1.525, 305.0], (-1.525, 305.0], ..., (1220.0, 1525.0], (1220.0, 1525.0], (1220.0, 1525.0], (1220.0, 1525.0], (1220.0, 1525.0]]
Length: 1526
Categories (5, interval[float64, right]): [(-1.525, 305.0] < (305.0, 610.0] < (610.0, 915.0] < (915.0, 1220.0] < (1220.0, 1525.0]]
df_slice = df[(df['BsmtFinSF2'] >= 305) & (df['BsmtFinSF2'] <= 610)]
df_slice
Building_Class	Zoning_Class	Lot_Extent	Lot_Size	Road_Type	Property_Shape	Land_Outline	Utility_Type	Lot_Configuration	Property_Slope	Neighborhood	Condition1	Condition2	House_Type	House_Design	Overall_Material	House_Condition	Construction_Year	Remodel_Year	Roof_Design	Roof_Quality	Exterior1st	Exterior2nd	Brick_Veneer_Type	Brick_Veneer_Area	Exterior_Material	Exterior_Condition	Foundation_Type	Basement_Height	Basement_Condition	Exposure_Level	BsmtFinType1	BsmtFinSF1	BsmtFinType2	BsmtFinSF2	BsmtUnfSF	Total_Basement_Area	Heating_Type	Heating_Quality	Air_Conditioning	Electrical_System	First_Floor_Area	Second_Floor_Area	LowQualFinSF	Grade_Living_Area	Underground_Full_Bathroom	Underground_Half_Bathroom	Full_Bathroom_Above_Grade	Half_Bathroom_Above_Grade	Bedroom_Above_Grade	Kitchen_Above_Grade	Kitchen_Quality	Rooms_Above_Grade	Functional_Rate	Fireplaces	Garage	Garage_Built_Year	Garage_Finish_Year	Garage_Size	Garage_Area	Garage_Quality	Garage_Condition	Pavedd_Drive	W_Deck_Area	Open_Lobby_Area	Enclosed_Lobby_Area	Three_Season_Lobby_Area	Screen_Lobby_Area	Pool_Area	Miscellaneous_Value	Month_Sold	Year_Sold	Sale_Type	Sale_Condition
Id																																																																										
27	20	RLD	60.0	7200.000000	Paved	Reg	Lvl	AllPub	C	GS	NAmes	Norm	Norm	1Fam	1Story	5	7	1951	2000	Gable	SS	Wd Sdng	Wd Sdng	None	0.0	TA	TA	CB	TA	TA	Mn	BLQ	234.0	Rec	486.0	180.0	900.0	GasA	TA	Y	SBrkr	900	0	0	900	0.0	1.0	1	0	3	1	Gd	5	TF	0	Detchd	2005.0	Unf	2.0	685.330199	TA	TA	Y	-4.995048	49.715385	-14.667328	0	0	0	0	5	2010	WD	Normal
44	20	RLD	NaN	9200.000000	Paved	IR1	Lvl	AllPub	CulDSac	GS	CollgCr	Norm	Norm	1Fam	1Story	5	6	1975	1980	Hip	SS	VinylSd	VinylSd	None	0.0	TA	TA	CB	Gd	TA	Av	LwQ	280.0	BLQ	491.0	167.0	938.0	GasA	TA	Y	SBrkr	938	0	0	938	1.0	0.0	1	0	3	1	TA	5	TF	0	Detchd	1977.0	Unf	1.0	491.660252	TA	TA	Y	94.883003	5.646183	-2.390123	0	0	0	0	7	2008	WD	Normal
45	20	RLD	70.0	7945.000000	Paved	Reg	Lvl	AllPub	I	GS	NAmes	Norm	Norm	1Fam	1Story	5	6	1959	1959	Gable	SS	BrkFace	Wd Sdng	None	0.0	TA	TA	CB	TA	TA	No	ALQ	179.0	BLQ	506.0	465.0	1150.0	GasA	Ex	Y	FuseA	1150	0	0	1150	1.0	0.0	1	0	3	1	TA	6	TF	0	Attchd	1959.0	RFn	1.0	110.174412	TA	TA	Y	-136.453742	-58.287397	-35.534555	0	0	0	0	5	2006	WD	Normal
74	20	RLD	85.0	10200.000000	Paved	Reg	Lvl	AllPub	I	GS	NAmes	Norm	Norm	1Fam	1Story	5	7	1954	2003	Gable	SS	Wd Sdng	Wd Sdng	BrkFace	104.0	TA	TA	CB	TA	TA	No	ALQ	320.0	BLQ	362.0	404.0	1086.0	GasA	Gd	Y	SBrkr	1086	0	0	1086	1.0	0.0	1	0	3	1	TA	6	TF	0	Attchd	1989.0	Unf	2.0	304.188412	TA	TA	Y	146.610151	32.292174	84.695712	0	0	0	0	5	2010	WD	Normal
174	20	RLD	80.0	10197.000000	Paved	IR1	Lvl	AllPub	I	GS	NAmes	Norm	Norm	1Fam	1Story	6	5	1961	1961	Gable	SS	WdShing	Wd Shng	BrkCmn	491.0	TA	TA	CB	TA	TA	No	ALQ	288.0	Rec	374.0	700.0	1362.0	GasA	TA	Y	SBrkr	1362	0	0	1362	1.0	0.0	1	1	3	1	TA	6	TF	1	Attchd	1961.0	Unf	2.0	897.401783	TA	TA	Y	252.994360	87.216016	-113.708008	0	0	0	0	6	2008	COD	Normal
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
2726	80	RLD	80.0	3790.765212	Paved	Reg	Lvl	AllPub	I	GS	NAmes	Norm	Norm	1Fam	SLvl	5	7	1967	1967	Gable	SS	MetalSd	MetalSd	BrkFace	140.0	TA	TA	PC	TA	TA	Av	ALQ	602.0	Rec	402.0	137.0	1141.0	GasA	Gd	Y	SBrkr	1141	0	0	1141	1.0	0.0	1	0	3	1	TA	6	TF	0	Attchd	1967.0	Unf	1.0	568.000000	TA	TA	Y	0.000000	78.000000	0.000000	0	0	0	0	7	2006	WD	Normal
2807	20	RLD	50.0	13314.527730	Paved	Reg	Lvl	AllPub	I	GS	SWISU	Norm	Norm	1Fam	1Story	7	5	2004	2004	Shed	SS	VinylSd	VinylSd	None	0.0	TA	TA	PC	Gd	Gd	Mn	GLQ	510.0	LwQ	373.0	190.0	1073.0	GasA	Ex	Y	SBrkr	1073	0	0	1073	1.0	0.0	2	0	2	1	TA	4	TF	0	Detchd	2004.0	Unf	1.0	246.000000	TA	TA	Y	0.000000	120.000000	0.000000	0	0	0	0	5	2006	WD	Normal
2844	80	RLD	42.0	8810.491032	Paved	IR1	Lvl	AllPub	CulDSac	GS	CollgCr	Norm	Norm	1Fam	SLvl	6	6	1978	1978	Gable	SS	HdBoard	HdBoard	BrkFace	123.0	TA	TA	CB	TA	Gd	Av	ALQ	595.0	LwQ	400.0	0.0	995.0	GasA	TA	Y	SBrkr	1282	0	0	1282	0.0	1.0	2	0	3	1	TA	6	TF	0	Detchd	1989.0	Unf	3.0	672.000000	Fa	TA	Y	386.000000	0.000000	0.000000	0	0	0	0	4	2006	WD	Normal
2859	70	RLD	67.0	7051.507926	Paved	Reg	Bnk	AllPub	I	GS	Edwards	Feedr	Norm	1Fam	2Story	4	6	1910	2000	Gable	SS	Plywood	Plywood	None	0.0	TA	Gd	CB	Gd	TA	No	Rec	173.0	BLQ	337.0	166.0	676.0	GasA	Gd	Y	SBrkr	760	676	0	1436	1.0	0.0	2	0	3	1	TA	6	MD1	0	Attchd	1950.0	Unf	2.0	528.000000	TA	TA	Y	147.000000	0.000000	0.000000	0	0	0	420	10	2006	WD	Normal
2912	20	RLD	80.0	17795.736390	Paved	Reg	Lvl	AllPub	I	MS	Mitchel	Norm	Norm	1Fam	1Story	5	5	1969	1979	Gable	SS	Plywood	Plywood	BrkFace	194.0	TA	TA	PC	TA	TA	Av	Rec	119.0	BLQ	344.0	641.0	1104.0	GasA	Fa	Y	SBrkr	1360	0	0	1360	1.0	0.0	1	0	3	1	TA	8	TF	1	Attchd	1969.0	RFn	1.0	336.000000	TA	TA	Y	160.000000	0.000000	0.000000	0	0	0	0	5	2006	WD	Normal
107 rows × 74 columns

Zoning_Col.at[333,'BsmtFinType2'] = df_slice['BsmtFinType2'].mode()[0] # replace NAN value of BsmtFinType2 by mode of buet ((305.0, 610.0)
Zoning_Col
Zoning_Class	Lot_Extent	Utility_Type	Exterior1st	Exterior2nd	Brick_Veneer_Type	Brick_Veneer_Area	Basement_Height	Basement_Condition	Exposure_Level	BsmtFinType2
Id											
8	RLD	NaN	AllPub	HdBoard	HdBoard	Stone	240.0	Gd	TA	Mn	NaN
13	RLD	NaN	AllPub	HdBoard	Plywood	None	0.0	TA	TA	No	NaN
15	RLD	NaN	AllPub	MetalSd	MetalSd	BrkFace	212.0	TA	TA	No	NaN
17	RLD	NaN	AllPub	Wd Sdng	Wd Sdng	BrkFace	180.0	TA	TA	No	NaN
18	RLD	72.0	AllPub	MetalSd	MetalSd	None	0.0	NaN	NaN	NaN	NaN
...	...	...	...	...	...	...	...	...	...	...	...
2901	RLD	NaN	AllPub	Plywood	Plywood	None	0.0	Gd	TA	Gd	NaN
2902	RLD	NaN	AllPub	VinylSd	VinylSd	None	0.0	Gd	TA	Av	NaN
2905	NaN	125.0	AllPub	CB	VinylSd	None	0.0	NaN	NaN	NaN	NaN
2909	RLD	NaN	AllPub	Plywood	Plywood	None	0.0	TA	TA	No	NaN
333	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	Rec
580 rows × 11 columns

Zoning_col = ['Zoning_Class', 'Lot_Extent', 'Utility_Type', 'Exterior1st', 'Exterior2nd',
       'Brick_Veneer_Type', 'Brick_Veneer_Area', 'Basement_Height']
Zoning_col = df[Zoning_col]
Zoning_col
Zoning_Class	Lot_Extent	Utility_Type	Exterior1st	Exterior2nd	Brick_Veneer_Type	Brick_Veneer_Area	Basement_Height
Id								
1	RLD	65.0	AllPub	VinylSd	VinylSd	BrkFace	196.0	Gd
2	RLD	80.0	AllPub	MetalSd	MetalSd	None	0.0	Gd
3	RLD	68.0	AllPub	VinylSd	VinylSd	BrkFace	162.0	Gd
4	RLD	60.0	AllPub	Wd Sdng	Wd Shng	None	0.0	TA
5	RLD	84.0	AllPub	VinylSd	VinylSd	BrkFace	350.0	Gd
...	...	...	...	...	...	...	...	...
2915	RMD	21.0	AllPub	CemntBd	CmentBd	None	0.0	TA
2916	RMD	21.0	AllPub	CemntBd	CmentBd	None	0.0	TA
2917	RLD	160.0	AllPub	VinylSd	VinylSd	None	0.0	TA
2918	RLD	62.0	AllPub	HdBoard	Wd Shng	None	0.0	Gd
2919	RLD	74.0	AllPub	HdBoard	HdBoard	BrkFace	94.0	Gd
2918 rows × 8 columns

Zoning_Col['Basement_Condition'] = Zoning_Col['Basement_Condition'].replace(np.nan, df['Basement_Condition'].mode()[0])
Zoning_Col['Basement_Height'] = Zoning_Col['Basement_Height'].replace(np.nan, df['Basement_Height'].mode()[0])
df.update(Zoning_Col)
Zoning_Col.isnull().sum()
Zoning_Class            5
Lot_Extent            487
Utility_Type            3
Exterior1st             2
Exterior2nd             2
Brick_Veneer_Type      25
Brick_Veneer_Area      24
Basement_Height         0
Basement_Condition      0
Exposure_Level         83
BsmtFinType2          579
dtype: int64
Handling missing value of Garage feature
df.columns[df.isnull().any()]
Index(['Zoning_Class', 'Lot_Extent', 'Utility_Type', 'Exterior1st',
       'Exterior2nd', 'Brick_Veneer_Type', 'Brick_Veneer_Area',
       'Exposure_Level', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'Total_Basement_Area', 'Electrical_System',
       'Underground_Full_Bathroom', 'Underground_Half_Bathroom',
       'Kitchen_Quality', 'Functional_Rate', 'Garage', 'Garage_Built_Year',
       'Garage_Finish_Year', 'Garage_Size', 'Garage_Area', 'Garage_Quality',
       'Garage_Condition', 'Sale_Type'],
      dtype='object')
Zoning_Col= ['Zoning_Class', 'Lot_Extent', 'Utility_Type', 'Garage_Finish_Year']
Zoning_Col= df[Zoning_Col]
Zoning_Col = Zoning_Col[Zoning_Col.isnull().any(axis=1)]
Zoning_Col
Zoning_Class	Lot_Extent	Utility_Type	Garage_Finish_Year
Id				
8	RLD	NaN	AllPub	RFn
13	RLD	NaN	AllPub	Unf
15	RLD	NaN	AllPub	RFn
17	RLD	NaN	AllPub	Fin
25	RLD	NaN	AllPub	Unf
...	...	...	...	...
2909	RLD	NaN	AllPub	Unf
2910	RMD	21.0	AllPub	NaN
2914	RMD	21.0	AllPub	NaN
2915	RMD	21.0	AllPub	NaN
2918	RLD	62.0	AllPub	NaN
637 rows × 4 columns

Zoning_Col.shape
(637, 4)
Zoning_Col_all_nan = Zoning_Col[(Zoning_Col.isnull() | Zoning_Col.isin([0])).all(1)]
Zoning_Col_all_nan.shape
(0, 4)
for i in Zoning_Col:
    if i in qual:
        Zoning_Col_all_nan[i] = Zoning_Col_all_nan[i].replace(np.nan, 'NA')
    else:
        Zoning_Col_all_nan[i] = Zoning_Col_all_nan[i].replace(np.nan, 0)
        
Zoning_Col.update(Zoning_Col_all_nan)
df.update(Zoning_Col_all_nan)
Zoning_Col = Zoning_Col[Zoning_Col.isnull().any(axis=1)]
Zoning_Col
Zoning_Class	Lot_Extent	Utility_Type	Garage_Finish_Year
Id				
8	RLD	NaN	AllPub	RFn
13	RLD	NaN	AllPub	Unf
15	RLD	NaN	AllPub	RFn
17	RLD	NaN	AllPub	Fin
25	RLD	NaN	AllPub	Unf
...	...	...	...	...
2909	RLD	NaN	AllPub	Unf
2910	RMD	21.0	AllPub	NaN
2914	RMD	21.0	AllPub	NaN
2915	RMD	21.0	AllPub	NaN
2918	RLD	62.0	AllPub	NaN
637 rows × 4 columns

for i in Zoning_Col:
    Zoning_Col[i] = Zoning_Col[i].replace(np.nan, df[df['Utility_Type'] == 'AllPub'][i].mode()[0])
Zoning_Col.isnull().any()
Zoning_Class          False
Lot_Extent            False
Utility_Type          False
Garage_Finish_Year    False
dtype: bool
df.update(Zoning_Col)
Handling missing value of remain feature
df.columns[df.isnull().any()]
Index(['Exterior1st', 'Exterior2nd', 'Brick_Veneer_Type', 'Brick_Veneer_Area',
       'Exposure_Level', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'Total_Basement_Area', 'Electrical_System',
       'Underground_Full_Bathroom', 'Underground_Half_Bathroom',
       'Kitchen_Quality', 'Functional_Rate', 'Garage', 'Garage_Built_Year',
       'Garage_Size', 'Garage_Area', 'Garage_Quality', 'Garage_Condition',
       'Sale_Type'],
      dtype='object')
df.columns[df.isnull().any()]
Index(['Exterior1st', 'Exterior2nd', 'Brick_Veneer_Type', 'Brick_Veneer_Area',
       'Exposure_Level', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'Total_Basement_Area', 'Electrical_System',
       'Underground_Full_Bathroom', 'Underground_Half_Bathroom',
       'Kitchen_Quality', 'Functional_Rate', 'Garage', 'Garage_Built_Year',
       'Garage_Size', 'Garage_Area', 'Garage_Quality', 'Garage_Condition',
       'Sale_Type'],
      dtype='object')
df[df['Brick_Veneer_Area'].isnull() == True]['Brick_Veneer_Type'].unique()
array([nan], dtype=object)
df.loc[(df['Brick_Veneer_Type'] == 'None') & (df['Brick_Veneer_Area'].isnull() == True), 'Brick_Veneer_Area'] = 0
df.isnull().sum()/df.shape[0] * 100
Building_Class               0.000000
Zoning_Class                 0.000000
Lot_Extent                   0.000000
Lot_Size                     0.000000
Road_Type                    0.000000
Property_Shape               0.000000
Land_Outline                 0.000000
Utility_Type                 0.000000
Lot_Configuration            0.000000
Property_Slope               0.000000
Neighborhood                 0.000000
Condition1                   0.000000
Condition2                   0.000000
House_Type                   0.000000
House_Design                 0.000000
Overall_Material             0.000000
House_Condition              0.000000
Construction_Year            0.000000
Remodel_Year                 0.000000
Roof_Design                  0.000000
Roof_Quality                 0.000000
Exterior1st                  0.034270
Exterior2nd                  0.034270
Brick_Veneer_Type            0.822481
Brick_Veneer_Area            0.788211
Exterior_Material            0.000000
Exterior_Condition           0.000000
Foundation_Type              0.000000
Basement_Height              0.000000
Basement_Condition           0.000000
Exposure_Level               2.810144
BsmtFinType1                 2.707334
BsmtFinSF1                   0.034270
BsmtFinType2                 2.707334
BsmtFinSF2                   0.034270
BsmtUnfSF                    0.034270
Total_Basement_Area          0.034270
Heating_Type                 0.000000
Heating_Quality              0.000000
Air_Conditioning             0.000000
Electrical_System            0.034270
First_Floor_Area             0.000000
Second_Floor_Area            0.000000
LowQualFinSF                 0.000000
Grade_Living_Area            0.000000
Underground_Full_Bathroom    0.068540
Underground_Half_Bathroom    0.068540
Full_Bathroom_Above_Grade    0.000000
Half_Bathroom_Above_Grade    0.000000
Bedroom_Above_Grade          0.000000
Kitchen_Above_Grade          0.000000
Kitchen_Quality              0.034270
Rooms_Above_Grade            0.000000
Functional_Rate              0.068540
Fireplaces                   0.000000
Garage                       5.380398
Garage_Built_Year            5.448938
Garage_Finish_Year           0.000000
Garage_Size                  0.034270
Garage_Area                  0.034270
Garage_Quality               5.448938
Garage_Condition             5.448938
Pavedd_Drive                 0.000000
W_Deck_Area                  0.000000
Open_Lobby_Area              0.000000
Enclosed_Lobby_Area          0.000000
Three_Season_Lobby_Area      0.000000
Screen_Lobby_Area            0.000000
Pool_Area                    0.000000
Miscellaneous_Value          0.000000
Month_Sold                   0.000000
Year_Sold                    0.000000
Sale_Type                    0.034270
Sale_Condition               0.000000
dtype: float64
Handling missing value of LotFrontage feature
lotconfig = ['Corner', 'Inside', 'CulDSac', 'FR2', 'FR3']
for i in lotconfig:
    df['LotFrontage'] = pd.np.where((df['LotFrontage'].isnull() == True) & (df['LotConfig'] == i) , df[df['LotConfig'] == i] ['LotFrontage'].mean(), df['LotFrontage'])
df.isnull().sum()
Building_Class                 0
Zoning_Class                   0
Lot_Extent                     0
Lot_Size                       0
Road_Type                      0
Property_Shape                 0
Land_Outline                   0
Utility_Type                   0
Lot_Configuration              0
Property_Slope                 0
Neighborhood                   0
Condition1                     0
Condition2                     0
House_Type                     0
House_Design                   0
Overall_Material               0
House_Condition                0
Construction_Year              0
Remodel_Year                   0
Roof_Design                    0
Roof_Quality                   0
Exterior1st                    1
Exterior2nd                    1
Brick_Veneer_Type             24
Brick_Veneer_Area             23
Exterior_Material              0
Exterior_Condition             0
Foundation_Type                0
Basement_Height                0
Basement_Condition             0
Exposure_Level                82
BsmtFinType1                  79
BsmtFinSF1                     1
BsmtFinType2                  79
BsmtFinSF2                     1
BsmtUnfSF                      1
Total_Basement_Area            1
Heating_Type                   0
Heating_Quality                0
Air_Conditioning               0
Electrical_System              1
First_Floor_Area               0
Second_Floor_Area              0
LowQualFinSF                   0
Grade_Living_Area              0
Underground_Full_Bathroom      2
Underground_Half_Bathroom      2
Full_Bathroom_Above_Grade      0
Half_Bathroom_Above_Grade      0
Bedroom_Above_Grade            0
Kitchen_Above_Grade            0
Kitchen_Quality                1
Rooms_Above_Grade              0
Functional_Rate                2
Fireplaces                     0
Garage                       157
Garage_Built_Year            159
Garage_Finish_Year             0
Garage_Size                    1
Garage_Area                    1
Garage_Quality               159
Garage_Condition             159
Pavedd_Drive                   0
W_Deck_Area                    0
Open_Lobby_Area                0
Enclosed_Lobby_Area            0
Three_Season_Lobby_Area        0
Screen_Lobby_Area              0
Pool_Area                      0
Miscellaneous_Value            0
Month_Sold                     0
Year_Sold                      0
Sale_Type                      1
Sale_Condition                 0
dtype: int64
Feature Transformation
df.columns
Index(['Building_Class', 'Zoning_Class', 'Lot_Extent', 'Lot_Size', 'Road_Type',
       'Property_Shape', 'Land_Outline', 'Utility_Type', 'Lot_Configuration',
       'Property_Slope', 'Neighborhood', 'Condition1', 'Condition2',
       'House_Type', 'House_Design', 'Overall_Material', 'House_Condition',
       'Construction_Year', 'Remodel_Year', 'Roof_Design', 'Roof_Quality',
       'Exterior1st', 'Exterior2nd', 'Brick_Veneer_Type', 'Brick_Veneer_Area',
       'Exterior_Material', 'Exterior_Condition', 'Foundation_Type',
       'Basement_Height', 'Basement_Condition', 'Exposure_Level',
       'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
       'Total_Basement_Area', 'Heating_Type', 'Heating_Quality',
       'Air_Conditioning', 'Electrical_System', 'First_Floor_Area',
       'Second_Floor_Area', 'LowQualFinSF', 'Grade_Living_Area',
       'Underground_Full_Bathroom', 'Underground_Half_Bathroom',
       'Full_Bathroom_Above_Grade', 'Half_Bathroom_Above_Grade',
       'Bedroom_Above_Grade', 'Kitchen_Above_Grade', 'Kitchen_Quality',
       'Rooms_Above_Grade', 'Functional_Rate', 'Fireplaces', 'Garage',
       'Garage_Built_Year', 'Garage_Finish_Year', 'Garage_Size', 'Garage_Area',
       'Garage_Quality', 'Garage_Condition', 'Pavedd_Drive', 'W_Deck_Area',
       'Open_Lobby_Area', 'Enclosed_Lobby_Area', 'Three_Season_Lobby_Area',
       'Screen_Lobby_Area', 'Pool_Area', 'Miscellaneous_Value', 'Month_Sold',
       'Year_Sold', 'Sale_Type', 'Sale_Condition'],
      dtype='object')
# converting columns in str which have categorical nature but in int64
col_dtype_convert = ['Garage','Garage_Finish_Year', 'Garage_Built_Year', 'Year_Sold']
for i in col_dtype_convert:
    df[i] = df[i].astype(str)
df['Month_Sold'].unique() # Month_Sold = Month of sold
array([ 2,  5,  9, 12, 10,  8, 11,  4,  1,  7,  3,  6], dtype=int64)
# conver in month abbrevation
import calendar
df['Month_Sold'] = df['Month_Sold'].apply(lambda x : calendar.month_abbr[x])
df['Month_Sold'].unique()
array(['Feb', 'May', 'Sep', 'Dec', 'Oct', 'Aug', 'Nov', 'Apr', 'Jan',
       'Jul', 'Mar', 'Jun'], dtype=object)
quan= list(df.loc[:, df.dtypes != 'object'].columns.values)
quan
['Building_Class',
 'Lot_Extent',
 'Lot_Size',
 'Overall_Material',
 'House_Condition',
 'Construction_Year',
 'Remodel_Year',
 'Brick_Veneer_Area',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'Total_Basement_Area',
 'First_Floor_Area',
 'Second_Floor_Area',
 'LowQualFinSF',
 'Grade_Living_Area',
 'Underground_Full_Bathroom',
 'Underground_Half_Bathroom',
 'Full_Bathroom_Above_Grade',
 'Half_Bathroom_Above_Grade',
 'Bedroom_Above_Grade',
 'Kitchen_Above_Grade',
 'Rooms_Above_Grade',
 'Fireplaces',
 'W_Deck_Area',
 'Open_Lobby_Area',
 'Enclosed_Lobby_Area',
 'Three_Season_Lobby_Area',
 'Screen_Lobby_Area',
 'Pool_Area',
 'Miscellaneous_Value']
len(quan)
31
obj_col = list(df.loc[:, df.dtypes == 'object'].columns.values)
obj_col
['Zoning_Class',
 'Road_Type',
 'Property_Shape',
 'Land_Outline',
 'Utility_Type',
 'Lot_Configuration',
 'Property_Slope',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'House_Type',
 'House_Design',
 'Roof_Design',
 'Roof_Quality',
 'Exterior1st',
 'Exterior2nd',
 'Brick_Veneer_Type',
 'Exterior_Material',
 'Exterior_Condition',
 'Foundation_Type',
 'Basement_Height',
 'Basement_Condition',
 'Exposure_Level',
 'BsmtFinType1',
 'BsmtFinType2',
 'Heating_Type',
 'Heating_Quality',
 'Air_Conditioning',
 'Electrical_System',
 'Kitchen_Quality',
 'Functional_Rate',
 'Garage',
 'Garage_Built_Year',
 'Garage_Finish_Year',
 'Garage_Size',
 'Garage_Area',
 'Garage_Quality',
 'Garage_Condition',
 'Pavedd_Drive',
 'Month_Sold',
 'Year_Sold',
 'Sale_Type',
 'Sale_Condition']
Conver categorical code into order
from pandas.api.types import CategoricalDtype
df['Basement_Condition'] = df['Basement_Condition'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['Basement_Condition'].unique()
array([-1], dtype=int8)
df['Exposure_Level'] = df['Exposure_Level'].astype(CategoricalDtype(categories=['NA', 'Mn', 'Av', 'Gd'], ordered = True)).cat.codes
df['Exposure_Level'].unique()
array([-1], dtype=int8)
df['BsmtFinType1'] = df['BsmtFinType1'].astype(CategoricalDtype(categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ','ALQ', 'GLQ'], ordered = True)).cat.codes
df['BsmtFinType2'] = df['BsmtFinType2'].astype(CategoricalDtype(categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ','ALQ', 'GLQ'], ordered = True)).cat.codes
df['Exterior_Material'] = df['Exterior_Material'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['Exterior_Condition'] = df['Exterior_Condition'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['Foundation_Type'] = df['Foundation_Type'].astype(CategoricalDtype(categories=['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod','Min2','Min1', 'Typ'], ordered = True)).cat.codes
df['Garage_Condition'] = df['Garage_Condition'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['Garage_Quality'] = df['Garage_Quality'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['Garage_Finish_Year'] = df['Garage_Finish_Year'].astype(CategoricalDtype(categories=['NA', 'Unf', 'RFn', 'Fin'], ordered = True)).cat.codes
df['Heating_Quality'] = df['Heating_Quality'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['Kitchen_Quality'] = df['Kitchen_Quality'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['Pavedd_Drive'] = df['Pavedd_Drive'].astype(CategoricalDtype(categories=['N', 'P', 'Y'], ordered = True)).cat.codes
df['Utility_Type'] = df['Utility_Type'].astype(CategoricalDtype(categories=['ELO', 'NASeWa', 'NASeWr', 'AllPub'], ordered = True)).cat.codes
df['Utility_Type'].unique()
array([ 3, -1], dtype=int8)
Show skewness of feature with distplot
Sale_Price = np.log(train['Sale_Price'] + 1)
obj_feat = list(df.loc[:,df.dtypes == 'object'].columns.values)
len(obj_feat)
29
# dummy varaibale
dummy_drop = []
clean_df = df
for i in obj_feat:
    dummy_drop += [i + '_' + str(df[i].unique()[-1])]

df = pd.get_dummies(df, columns = obj_feat)
df = df.drop(dummy_drop, axis = 1)
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
File C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\indexes\base.py:3621, in Index.get_loc(self, key, method, tolerance)
   3620 try:
-> 3621     return self._engine.get_loc(casted_key)
   3622 except KeyError as err:

File C:\ProgramData\Anaconda3\lib\site-packages\pandas\_libs\index.pyx:136, in pandas._libs.index.IndexEngine.get_loc()

File C:\ProgramData\Anaconda3\lib\site-packages\pandas\_libs\index.pyx:163, in pandas._libs.index.IndexEngine.get_loc()

File pandas\_libs\hashtable_class_helper.pxi:5198, in pandas._libs.hashtable.PyObjectHashTable.get_item()

File pandas\_libs\hashtable_class_helper.pxi:5206, in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'Zoning_Class'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
Input In [206], in <cell line: 3>()
      2 clean_df = df
      3 for i in obj_feat:
----> 4     dummy_drop += [i + '_' + str(df[i].unique()[-1])]
      6 df = pd.get_dummies(df, columns = obj_feat)
      7 df = df.drop(Zoning_Class, axis = 1)

File C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\frame.py:3505, in DataFrame.__getitem__(self, key)
   3503 if self.columns.nlevels > 1:
   3504     return self._getitem_multilevel(key)
-> 3505 indexer = self.columns.get_loc(key)
   3506 if is_integer(indexer):
   3507     indexer = [indexer]

File C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\indexes\base.py:3623, in Index.get_loc(self, key, method, tolerance)
   3621     return self._engine.get_loc(casted_key)
   3622 except KeyError as err:
-> 3623     raise KeyError(key) from err
   3624 except TypeError:
   3625     # If we have a listlike key, _check_indexing_error will raise
   3626     #  InvalidIndexError. Otherwise we fall through and re-raise
   3627     #  the TypeError.
   3628     self._check_indexing_error(key)

KeyError: 'Zoning_Class'
df.shape
(2918, 2274)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(df)
df = scaler.transform(df)
train_len = len(train)
X_train = df[:train_len]
X_test = df[train_len:]
y_train = Sale_Price

print(X_train.shape)
print(X_test.shape)
print(len(y_train))
(1459, 2274)
(1459, 2274)
1459
Cross Validation
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

def test_model(model, X_train=X_train, y_train=y_train):
    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)
    score = [r2_val_score.mean()]
    return score
Linear Regression
import sklearn.linear_model as linear_model
LR = linear_model.LinearRegression()
test_model(LR)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
3 fits failed out of a total of 3.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_base.py", line 662, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
[nan]
# Cross validation
cross_validation = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = 10)
print("Cross validation accuracy of LR model = ", cross_validation)
print("\nCross validation mean accuracy of LR model = ", cross_validation.mean())
Cross validation accuracy of LR model =  [nan nan nan nan nan nan nan nan nan nan]

Cross validation mean accuracy of LR model =  nan
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
10 fits failed out of a total of 10.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
10 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_base.py", line 662, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
rdg = linear_model.Ridge()
test_model(rdg)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
3 fits failed out of a total of 3.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py", line 1003, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
[nan]
lasso = linear_model.Lasso(alpha=1e-4)
test_model(lasso)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
3 fits failed out of a total of 3.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_coordinate_descent.py", line 935, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
[nan]
Support Vector Machine
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
test_model(svr_reg)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
3 fits failed out of a total of 3.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\svm\_base.py", line 190, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
[nan]
Decision Tree Regresso
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=21)
test_model(dt_reg)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
3 fits failed out of a total of 3.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\tree\_classes.py", line 1315, in fit
    super().fit(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\tree\_classes.py", line 165, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 578, in _validate_data
    X = check_array(X, **check_X_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float32').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
[nan]
Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 1000, random_state=51)
test_model(rf_reg)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
3 fits failed out of a total of 3.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\ensemble\_forest.py", line 327, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float32').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
[nan]
Bagging & Boosting
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
br_reg = BaggingRegressor(n_estimators=1000, random_state=51)
gbr_reg = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, loss='ls', random_state=51)
test_model(gbr_reg)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
3 fits failed out of a total of 3.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\ensemble\_gb.py", line 486, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float32').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
[nan]
Feature Selection / Engineering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('Propert_Price_Train.csv')
df = pd.read_csv('Property_Price_Test.csv')
train = pd.read_csv('Propert_Price_Train.csv')
test = pd.read_csv('Property_Price_Test.csv')
df
Id	Building_Class	Zoning_Class	Lot_Extent	Lot_Size	Road_Type	Lane_Type	Property_Shape	Land_Outline	Utility_Type	Lot_Configuration	Property_Slope	Neighborhood	Condition1	Condition2	House_Type	House_Design	Overall_Material	House_Condition	Construction_Year	Remodel_Year	Roof_Design	Roof_Quality	Exterior1st	Exterior2nd	Brick_Veneer_Type	Brick_Veneer_Area	Exterior_Material	Exterior_Condition	Foundation_Type	Basement_Height	Basement_Condition	Exposure_Level	BsmtFinType1	BsmtFinSF1	BsmtFinType2	BsmtFinSF2	BsmtUnfSF	Total_Basement_Area	Heating_Type	Heating_Quality	Air_Conditioning	Electrical_System	First_Floor_Area	Second_Floor_Area	LowQualFinSF	Grade_Living_Area	Underground_Full_Bathroom	Underground_Half_Bathroom	Full_Bathroom_Above_Grade	Half_Bathroom_Above_Grade	Bedroom_Above_Grade	Kitchen_Above_Grade	Kitchen_Quality	Rooms_Above_Grade	Functional_Rate	Fireplaces	Fireplace_Quality	Garage	Garage_Built_Year	Garage_Finish_Year	Garage_Size	Garage_Area	Garage_Quality	Garage_Condition	Pavedd_Drive	W_Deck_Area	Open_Lobby_Area	Enclosed_Lobby_Area	Three_Season_Lobby_Area	Screen_Lobby_Area	Pool_Area	Pool_Quality	Fence_Quality	Miscellaneous_Feature	Miscellaneous_Value	Month_Sold	Year_Sold	Sale_Type	Sale_Condition
0	1461	20	RHD	80.0	16104.819760	Paved	NaN	Reg	Lvl	AllPub	I	GS	NAmes	Feedr	Norm	1Fam	1Story	5	6	1961	1961	Gable	SS	VinylSd	VinylSd	None	0.0	TA	TA	CB	TA	TA	No	Rec	468.0	LwQ	144.0	270.0	882.0	GasA	TA	Y	SBrkr	896	0	0	896	0.0	0.0	1	0	2	1	TA	5	TF	0	NaN	Attchd	1961.0	Unf	1.0	730.0	TA	TA	Y	140	0	0	0	120	0	NaN	MnPrv	NaN	0	6	2010	WD	Normal
1	1462	20	RLD	81.0	15639.150810	Paved	NaN	IR1	Lvl	AllPub	C	GS	NAmes	Norm	Norm	1Fam	1Story	6	6	1958	1958	Hip	SS	Wd Sdng	Wd Sdng	BrkFace	108.0	TA	TA	CB	TA	TA	No	ALQ	923.0	Unf	0.0	406.0	1329.0	GasA	TA	Y	SBrkr	1329	0	0	1329	0.0	0.0	1	1	3	1	Gd	6	TF	0	NaN	Attchd	1958.0	Unf	1.0	312.0	TA	TA	Y	393	36	0	0	0	0	NaN	NaN	Gar2	12500	6	2010	WD	Normal
2	1463	60	RLD	74.0	3849.428920	Paved	NaN	IR1	Lvl	AllPub	I	GS	Gilbert	Norm	Norm	1Fam	2Story	5	5	1997	1998	Gable	SS	VinylSd	VinylSd	None	0.0	TA	TA	PC	Gd	TA	No	GLQ	791.0	Unf	0.0	137.0	928.0	GasA	Gd	Y	SBrkr	928	701	0	1629	0.0	0.0	2	1	3	1	TA	6	TF	1	TA	Attchd	1997.0	Fin	2.0	482.0	TA	TA	Y	212	34	0	0	0	0	NaN	MnPrv	NaN	0	3	2010	WD	Normal
3	1464	60	RLD	78.0	4955.447942	Paved	NaN	IR1	Lvl	AllPub	I	GS	Gilbert	Norm	Norm	1Fam	2Story	6	6	1998	1998	Gable	SS	VinylSd	VinylSd	BrkFace	20.0	TA	TA	PC	TA	TA	No	GLQ	602.0	Unf	0.0	324.0	926.0	GasA	Ex	Y	SBrkr	926	678	0	1604	0.0	0.0	2	1	3	1	Gd	7	TF	1	Gd	Attchd	1998.0	Fin	2.0	470.0	TA	TA	Y	360	36	0	0	0	0	NaN	NaN	NaN	0	6	2010	WD	Normal
4	1465	120	RLD	43.0	3046.604942	Paved	NaN	IR1	HLS	AllPub	I	GS	StoneBr	Norm	Norm	TwnhsE	1Story	8	5	1992	1992	Gable	SS	HdBoard	HdBoard	None	0.0	Gd	TA	PC	Gd	TA	No	ALQ	263.0	Unf	0.0	1017.0	1280.0	GasA	Ex	Y	SBrkr	1280	0	0	1280	0.0	0.0	2	0	2	1	Gd	5	TF	0	NaN	Attchd	1992.0	RFn	2.0	506.0	TA	TA	Y	0	82	0	0	144	0	NaN	NaN	NaN	0	1	2010	WD	Normal
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
1454	2915	160	RMD	21.0	14584.838440	Paved	NaN	Reg	Lvl	AllPub	I	GS	MeadowV	NoRMD	NoRMD	Twnhs	2Story	4	7	1970	1970	Gable	SS	CemntBd	CmentBd	None	0.0	TA	TA	CB	TA	TA	No	Unf	0.0	Unf	0.0	546.0	546.0	GasA	Gd	Y	SBrkr	546	546	0	1092	0.0	0.0	1	1	3	1	TA	5	TF	0	NaN	NaN	NaN	NaN	0.0	0.0	NaN	NaN	Y	0	0	0	0	0	0	NaN	NaN	NaN	0	6	2006	WD	NoRMDal
1455	2916	160	RMD	21.0	8072.991379	Paved	NaN	Reg	Lvl	AllPub	I	GS	MeadowV	NoRMD	NoRMD	TwnhsE	2Story	4	5	1970	1970	Gable	SS	CemntBd	CmentBd	None	0.0	TA	TA	CB	TA	TA	No	Rec	252.0	Unf	0.0	294.0	546.0	GasA	TA	Y	SBrkr	546	546	0	1092	0.0	0.0	1	1	3	1	TA	6	TF	0	NaN	CarPort	1970.0	Unf	1.0	286.0	TA	TA	Y	0	24	0	0	0	0	NaN	NaN	NaN	0	4	2006	WD	AbnoRMDl
1456	2917	20	RLD	160.0	7367.775348	Paved	NaN	Reg	Lvl	AllPub	I	GS	Mitchel	Norm	Norm	1Fam	1Story	5	7	1960	1996	Gable	SS	VinylSd	VinylSd	None	0.0	TA	TA	CB	TA	TA	No	ALQ	1224.0	Unf	0.0	0.0	1224.0	GasA	Ex	Y	SBrkr	1224	0	0	1224	1.0	0.0	1	0	4	1	TA	7	TF	1	TA	Detchd	1960.0	Unf	2.0	576.0	TA	TA	Y	474	0	0	0	0	0	NaN	NaN	NaN	0	9	2006	WD	Abnorml
1457	2918	85	RLD	62.0	2203.135444	Paved	NaN	Reg	Lvl	AllPub	I	GS	Mitchel	Norm	Norm	1Fam	SFoyer	5	5	1992	1992	Gable	SS	HdBoard	Wd Shng	None	0.0	TA	TA	PC	Gd	TA	Av	GLQ	337.0	Unf	0.0	575.0	912.0	GasA	TA	Y	SBrkr	970	0	0	970	0.0	1.0	1	0	3	1	TA	6	TF	0	NaN	NaN	NaN	NaN	0.0	0.0	NaN	NaN	Y	80	32	0	0	0	0	NaN	MnPrv	Shed	700	7	2006	WD	Normal
1458	2919	60	RLD	74.0	6253.431852	Paved	NaN	Reg	Lvl	AllPub	I	MS	Mitchel	Norm	Norm	1Fam	2Story	7	5	1993	1994	Gable	SS	HdBoard	HdBoard	BrkFace	94.0	TA	TA	PC	Gd	TA	Av	LwQ	758.0	Unf	0.0	238.0	996.0	GasA	Ex	Y	SBrkr	996	1004	0	2000	0.0	0.0	2	1	3	1	TA	9	TF	1	TA	Attchd	1993.0	Fin	3.0	650.0	TA	TA	Y	190	48	0	0	0	0	NaN	NaN	NaN	0	11	2006	WD	Normal
1459 rows × 80 columns

Drop Feature
df = df.drop(['Year_Sold',
 'LowQualFinSF',
 'Miscellaneous_Value',
 'BsmtFinSF2',
 'Month_Sold'],axis=1)
quan = list(df.loc[:,df.dtypes != 'object'].columns.values)
quan
['Id',
 'Building_Class',
 'Lot_Extent',
 'Lot_Size',
 'Overall_Material',
 'House_Condition',
 'Construction_Year',
 'Remodel_Year',
 'Brick_Veneer_Area',
 'BsmtFinSF1',
 'BsmtUnfSF',
 'Total_Basement_Area',
 'First_Floor_Area',
 'Second_Floor_Area',
 'Grade_Living_Area',
 'Underground_Full_Bathroom',
 'Underground_Half_Bathroom',
 'Full_Bathroom_Above_Grade',
 'Half_Bathroom_Above_Grade',
 'Bedroom_Above_Grade',
 'Kitchen_Above_Grade',
 'Rooms_Above_Grade',
 'Fireplaces',
 'Garage_Built_Year',
 'Garage_Size',
 'Garage_Area',
 'W_Deck_Area',
 'Open_Lobby_Area',
 'Enclosed_Lobby_Area',
 'Three_Season_Lobby_Area',
 'Screen_Lobby_Area',
 'Pool_Area']
obj_feat = list(df.loc[:, df.dtypes == 'object'].columns.values)
print(len(obj_feat))

obj_feat
43
['Zoning_Class',
 'Road_Type',
 'Lane_Type',
 'Property_Shape',
 'Land_Outline',
 'Utility_Type',
 'Lot_Configuration',
 'Property_Slope',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'House_Type',
 'House_Design',
 'Roof_Design',
 'Roof_Quality',
 'Exterior1st',
 'Exterior2nd',
 'Brick_Veneer_Type',
 'Exterior_Material',
 'Exterior_Condition',
 'Foundation_Type',
 'Basement_Height',
 'Basement_Condition',
 'Exposure_Level',
 'BsmtFinType1',
 'BsmtFinType2',
 'Heating_Type',
 'Heating_Quality',
 'Air_Conditioning',
 'Electrical_System',
 'Kitchen_Quality',
 'Functional_Rate',
 'Fireplace_Quality',
 'Garage',
 'Garage_Finish_Year',
 'Garage_Quality',
 'Garage_Condition',
 'Pavedd_Drive',
 'Pool_Quality',
 'Fence_Quality',
 'Miscellaneous_Feature',
 'Sale_Type',
 'Sale_Condition']
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(df)
df = scaler.transform(df)
train_len = len(train)
X_train = df[:train_len]
X_test = df[train_len:]
y_train = Sale_Price

print("Shape of X_train: ", len(X_train))
print("Shape of X_test: ", len(X_test))
print("Shape of y_train: ", len(y_train))
Shape of X_train:  1459
Shape of X_test:  0
Shape of y_train:  1459
Cross Validation
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

def test_model(model, X_train=X_train, y_train=y_train):
    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)
    score = [r2_val_score.mean()]
    return score
Linear Model
import sklearn.linear_model as linear_model
LR = linear_model.LinearRegression()
test_model(LR)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
3 fits failed out of a total of 3.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_base.py", line 662, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
[nan]
rdg = linear_model.Ridge()
test_model(rdg)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
3 fits failed out of a total of 3.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py", line 1003, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
[nan]
lasso = linear_model.Lasso(alpha=1e-4)
test_model(lasso)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
3 fits failed out of a total of 3.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_coordinate_descent.py", line 935, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
[nan]
Support Vector Machine
from sklearn.svm import SVR
svr = SVR(kernel='rbf')
test_model(svr)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
3 fits failed out of a total of 3.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\svm\_base.py", line 190, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
[nan]
Svm Hyper Parameter Tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
params = {'kernel': ['rbf'],
         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
         'C': [0.1, 1, 10, 100, 1000],
         'epsilon': [1, 0.2, 0.1, 0.01, 0.001, 0.0001]}
rand_search = RandomizedSearchCV(svr_reg, param_distributions=params, n_jobs=-1, cv=11)
rand_search.fit(X_train, y_train)
rand_search.best_score_
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
110 fits failed out of a total of 110.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
110 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\svm\_base.py", line 190, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:969: UserWarning: One or more of the test scores are non-finite: [nan nan nan nan nan nan nan nan nan nan]
  warnings.warn(
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Input In [260], in <cell line: 7>()
      2 params = {'kernel': ['rbf'],
      3          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
      4          'C': [0.1, 1, 10, 100, 1000],
      5          'epsilon': [1, 0.2, 0.1, 0.01, 0.001, 0.0001]}
      6 rand_search = RandomizedSearchCV(svr_reg, param_distributions=params, n_jobs=-1, cv=11)
----> 7 rand_search.fit(X_train, y_train)
      8 rand_search.best_score_

File C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:926, in BaseSearchCV.fit(self, X, y, groups, **fit_params)
    924 refit_start_time = time.time()
    925 if y is not None:
--> 926     self.best_estimator_.fit(X, y, **fit_params)
    927 else:
    928     self.best_estimator_.fit(X, **fit_params)

File C:\ProgramData\Anaconda3\lib\site-packages\sklearn\svm\_base.py:190, in BaseLibSVM.fit(self, X, y, sample_weight)
    188     check_consistent_length(X, y)
    189 else:
--> 190     X, y = self._validate_data(
    191         X,
    192         y,
    193         dtype=np.float64,
    194         order="C",
    195         accept_sparse="csr",
    196         accept_large_sparse=False,
    197     )
    199 y = self._validate_targets(y)
    201 sample_weight = np.asarray(
    202     [] if sample_weight is None else sample_weight, dtype=np.float64
    203 )

File C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py:581, in BaseEstimator._validate_data(self, X, y, reset, validate_separately, **check_params)
    579         y = check_array(y, **check_y_params)
    580     else:
--> 581         X, y = check_X_y(X, y, **check_params)
    582     out = X, y
    584 if not no_val_X and check_params.get("ensure_2d", True):

File C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py:964, in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
    961 if y is None:
    962     raise ValueError("y cannot be None")
--> 964 X = check_array(
    965     X,
    966     accept_sparse=accept_sparse,
    967     accept_large_sparse=accept_large_sparse,
    968     dtype=dtype,
    969     order=order,
    970     copy=copy,
    971     force_all_finite=force_all_finite,
    972     ensure_2d=ensure_2d,
    973     allow_nd=allow_nd,
    974     ensure_min_samples=ensure_min_samples,
    975     ensure_min_features=ensure_min_features,
    976     estimator=estimator,
    977 )
    979 y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric)
    981 check_consistent_length(X, y)

File C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py:800, in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
    794         raise ValueError(
    795             "Found array with dim %d. %s expected <= 2."
    796             % (array.ndim, estimator_name)
    797         )
    799     if force_all_finite:
--> 800         _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
    802 if ensure_min_samples > 0:
    803     n_samples = _num_samples(array)

File C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py:114, in _assert_all_finite(X, allow_nan, msg_dtype)
    107     if (
    108         allow_nan
    109         and np.isinf(X).any()
    110         or not allow_nan
    111         and not np.isfinite(X).all()
    112     ):
    113         type_err = "infinity" if allow_nan else "NaN, infinity"
--> 114         raise ValueError(
    115             msg_err.format(
    116                 type_err, msg_dtype if msg_dtype is not None else X.dtype
    117             )
    118         )
    119 # for object dtype data, we only check for NaNs (GH-13254)
    120 elif X.dtype == np.dtype("object") and not allow_nan:

ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
rand_search.best_estimator_
SVR(C=0.1, epsilon=0.2, gamma=0.0001)
svr_reg1=SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
test_model(svr_reg1)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
3 fits failed out of a total of 3.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\svm\_base.py", line 190, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
[nan]
svr_reg= SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.01, gamma=0.0001,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
test_model(svr_reg)
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
3 fits failed out of a total of 3.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\svm\_base.py", line 190, in fit
    X, y = self._validate_data(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py", line 581, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 964, in check_X_y
    X = check_array(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

  warnings.warn(some_fits_failed_message, FitFailedWarning)
[nan]
