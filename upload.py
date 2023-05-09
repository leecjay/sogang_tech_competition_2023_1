#uploading image to firebase storage
import pyrebase

#Input personal firebase information
config = {
    "apiKey": "",
    "authDomain": "",
    "databaseURL": "",
    "projectId": "",
    "storageBucket": "",
    "messagingSenderId": "",
    "appId": "",
    "measurementId": ""
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

path_on_cloud = "sample.jpg" #Name of image at the cloud that will be upload.
path_local = "C:/Users/.jpg" #Address of image in the local storage what you want to upload
storage.child(path_on_cloud).put(path_local)
