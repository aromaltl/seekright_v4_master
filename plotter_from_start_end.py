import googlemaps
import matplotlib.pyplot as plt
import polyline
import folium
import pandas as pd

gmaps = googlemaps.Client(key='AIzaSyCEETIYvSWfEJjx7YdauJJ3IPTOcctkRNw')
mymap = folium.Map(location=[23.2 ,45.2], zoom_start=6)
def plotter(i,j):
    direc=gmaps.directions(i,j)
    path_points = []
    # print(direc)
    polyline_points = direc[0]['overview_polyline']['points']

    # Decode the polyline points using the polyline library
    decoded_points = polyline.decode(polyline_points)

    # Extract latitude and longitude from decoded points
    # lats, lons = zip(*decoded_points)
    # for la,ln in zip(decoded_points):
        # print(la)
    lt,ln =zip(*decoded_points)
        # print(lt,ln)
    coordinates = [(lt,ln) for lt,ln in zip(lt,ln)]
    # print(coordinates)

    # mymap = folium.Map(location=[sum(x[0] for x in coordinates)/len(coordinates), 
    #                               sum(x[1] for x in coordinates)/len(coordinates)], zoom_start=15)

    # Add markers to the map
    # for coord in coordinates:
    #     folium.Marker(coord).add_to(mymap)
    folium.PolyLine(locations=coordinates, color='blue').add_to(mymap)
    # Save the map as an HTML file
    # mymap.save('map.html')
# print(path_points)

df=pd.read_csv("/home/sultan/Downloads/Internal_sheet.csv")
for key in set(df.columns) -set(["GPS Coordinates","Contracts"]):
    for i in range(0,len(df),2):
        if len(str(df[key][i])) > 5 and len(str(df[key][i+1])) > 5 :
            p=df[key][i].replace(" ",",").replace("N","").replace("E","")
            q=df[key][i+1].replace(" ",",").replace("N","").replace("E","")
            # try:
            try:
                plotter(p,q)
            except Exception as ex:
                print(ex)
                print(i,df["Contracts"][i],key,df[key][i],df[key][i+1])

mymap.save('map.html')