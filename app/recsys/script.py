from geopy import Nominatim
from math import *
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append('F:/KMIPN/ahahahay/api-web2/app')
from Model import IndoSbertModel
# from sentence_transformers import SentenceTransformer

# Dataset Pasal
baseURL= "https://drive.google.com/uc?export=download&id="
perpectPath = baseURL + "https://drive.google.com/file/d/1JSETqVkLXQNmQJAgLCgFbOYuQr1YDrdw/view?usp=sharing".split("/")[-2]
df = pd.read_csv(perpectPath)
to_replace = "Pewarisan Karena Kematian (Tidak Berlaku Bagi Golongan Timur Asing Bukan Tionghoa, Tetapi Berlaku Bagi Golongan Tionghoa)"
replace_with = "Pewarisan Karena Kematian"
df.Bab.replace(to_replace,replace_with,inplace=True)

#Dummy Pengacara
baseURL= "https://drive.google.com/uc?export=download&id="
perpectPath = baseURL + "https://drive.google.com/file/d/18lNIVOQrgV0zvKCL5Yb0JpxZTP8IwVWL/view?usp=drive_link".split("/")[-2]
pengacara = pd.read_json(perpectPath)



# intansiated model for text processing
model = IndoSbertModel()
model.fit(df,"Isi")

# BAGIAN 1 :  UTILITAS PERHITUNGAN JARAK
# suatu fungsi untuk mendaptkan lintang dan garis bujur dari suatu titik koordinat
def getLocationFromAndress(adress : str) -> dict[float,float]:
  loc = Nominatim(user_agent="GetLoc")
  adressLocation = loc.geocode(adress)
  return dict(latitude = adressLocation.latitude, longitude = adressLocation.longitude)

jarakPengacara = [getLocationFromAndress(a) for a in pengacara["lokasi"].unique()]
pemetaan = {pengacara.head()["lokasi"].values[i] : k for i,k in  enumerate(jarakPengacara)}

latitude = [pemetaan[a]["latitude"] for a in pengacara["lokasi"].values ]
longitude = [pemetaan[a]["longitude"] for a in pengacara["lokasi"].values ]
pengacara.insert(2,"latitude",latitude)
pengacara.insert(3,"longitude",longitude)

# fungsi haversine : suatu formula distance yang mempertimbangkan kelengkungan dari bumi
def haversine(lon1 : float,
              lat1 : float,
              lon2 : float,
              lat2 : float,
              on : str = "km") -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    result = c * r
    if on == "m":
      return result * 1000
    else:
      return result

# fungsi bantuan untuk menghitung bagaimana titik dari user di kalkulasi dengan jarak dari pengacara yang ada didalam database
def calculateDistance(targetX : dict[float,float],
                      targetY : list[dict[float,float]],
                      on : str ="km",
                      formula : "function" = haversine) -> list[float]:
  distance = []
  for i in targetY:
    distance.append(haversine(targetX["latitude"],targetX["longitude"],i["latitude"],i["longitude"],on = on))
  return distance

# BAGIAN 2 : UTILITAS TEXT EMBEDDINGS
# fungsi ini digunakan untuk mengestrak data dari bab yang relevan dari pasal, bab yang didapat akan dilakukan proses
# perhitungan kemiripan keahlian pengacara di fungsi selanjutnya
def getBabFromPasal(index : np.ndarray[int],
                    df : pd.DataFrame) -> list[str]:
  extractData = df.loc[index]
  posibleBab = []
  for i, j in extractData.iterrows():
    posibleBab.append(j.Bab)
  return posibleBab


# fungsi ini digunakan untuk melakukan kalkulasi terhadap similarity dari keahlian pengacara dengan pasal yang ada
def betweenBabAndLawyerSkill(listOfBab : list[str],
                             lawyer : pd.DataFrame) -> dict[str,float]:
  # memastikan tidak ada duplikat didalam listOfBab
  listOfBab = list(a.lower() for a in set(listOfBab))#
  listOfBabEmbedings = model.model.encode(listOfBab)

  # mendapatkan keahlian pengacara dan menjadikannya embbedings vektor
  keahlian = lawyer["keahlian"].apply(lambda x : x.lower()).values
  keahlianEmbeddings = model.model.encode(keahlian)

  # Iterasi terhadap dan kalkulasi
  score = []
  index = []
  for i,j in enumerate(keahlianEmbeddings):
    cosine = cosine_similarity(j.reshape(1,-1),listOfBabEmbedings)
    score.append(cosine[0][:5].mean())
    index.append(cosine.argsort(1)[0][:5])
  return score,index


# fungsi ini digunakan untuk menyama ratakan bobot nilai dari masing - masing variabel agar memiliki bobot yang sama
def minMaxScaller(data : np.ndarray[int | float]) -> np.ndarray[float]:
  dataResult = []
  for i in data:
    result = (i - data.min()) / (data.max() - data.min())
    dataResult.append(result)
  return dataResult

# fungsi yang akan melakukan kalkulasi dan mengembalikan nilai rekomendasi
def recomendationSystem(adress : str | dict[str,str],
                        question : str,
                        lawyer : pd.DataFrame = pengacara,
                        dataPasal : pd.DataFrame = df,
                        model : IndoSbertModel = model,
                        topk : int = 5,
                        getLocationFromAndress : "function" = getLocationFromAndress,
                        getBabFromPasal : "function" = getBabFromPasal,
                        distanceFormula : "function" = haversine,
                        scallerData : "function" = minMaxScaller,
                        ) -> pd.DataFrame:

  # get user location code from latitude and longitude
  latitudeUser = longitudeUser = None
  if type(adress) == str:
    userLocation = getLocationFromAndress(adress)
    latitudeUser = userLocation["latitude"]
    longitudeUser = userLocation["longitude"]
  elif type(adress) == dict:

    try :
      latitudeUser = adress["latitude"]
      longitudeUser = adress["longitude"]
    except Exception as e:
      raise ArgumentError("Wrong key, key must latitude and longitude got : {} "
      .format([a for a in adress.keys() if a not in ["latitude","longitude"]]))

  # get all location from lawyer -> just for info we use the index for recognize
  latitudeLawyer = lawyer["latitude"].values
  longitudeLawyer = lawyer["longitude"].values

  # transform the user question and get relevan pasal for user problem
  index,_ = model.fit_predict(question)
  babRelevanUser = getBabFromPasal(index,dataPasal)

  # get all value with Hoversine formula -> we calculated user location with laywer location in this case we define
  # let's say we already have pengacara lat and long
  allDistance =  np.array([distanceFormula(latitudeUser,longitudeUser,latitudeLawyer[a],longitudeLawyer[a]) for a in range(len(latitudeLawyer))])
  distanceRank = np.array([allDistance.max() - a for a in allDistance])


  # standart scallar or minmax scallar for make range data normazile in distanceRank
  scallingRank = scallerData(distanceRank)

  # get all similariies for relevant experties from lawyer with user problem
  similarityScore,_ = betweenBabAndLawyerSkill(babRelevanUser,lawyer)
  similarityScalling = scallerData(np.array(similarityScore))

  #  looping or something else that adding and calculate for similarity
  rankNScore = []



  for i in range(len(pengacara)):
    result = similarityScalling[i] + (scallingRank[i] * 0.05)
    rankNScore.append(result)

  return lawyer.loc[np.flip(np.array(rankNScore).argsort())[:topk]]
