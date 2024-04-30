
from django.shortcuts import redirect, render
from.models import reg,normal,xray
import cv2
import numpy as np
from django.shortcuts import render
import base64

from tensorflow.keras.models import load_model
from django.http import HttpResponse
from django.contrib.auth import logout

image_path=''
predictions=''
from tensorflow.keras.models import load_model

try:
    model = load_model('unet4_model.h5')
except Exception as e:
    pass
   

# Create your views here.
def index(request):
    return render(request,'index.html')

def register(request):
    if request.method=='POST':
        a=request.POST.get('name')
        b=request.POST.get('age')
        c=request.POST.get('uname')
        d=request.POST.get('email')
        e=request.POST.get('phone')
        f=request.POST.get('password')
        g=request.POST.get('cpassword')
        h=request.POST.get('gender')
        reg(name=a,age=b,uname=c,email=d,phone=e,password=f,cpassword=g,gender=h).save()
        return render(request,'index.html')
    else:
        return render(request,'register.html')
    
def login(request):
    if request.method=="POST":
        email=request.POST.get('email')
        # print(name)
        password = request.POST.get('password')
        # print('joy')
        cr = reg.objects.filter(email=email,password=password)
        if cr:
            userd =reg.objects.get(email=email,password=password)
            id=userd.id
            email=userd.email
            password=userd.password
            request.session['email']=email
            return render(request,'home.html')
        else:
            
            return render(request,'login.html')
    else:
        return render(request,'login.html')
    
def home(request):
    return render(request,'home.html')

def logoutv(request):
    logout(request)
    return redirect(index)

def profile(request):
    email=request.session['email']
    cr=reg.objects.get(email=email)
    if cr:
        user_info={
            'uname':cr.uname,
            'name':cr.name,
            'phone':cr.phone,
            'age':cr.age,
            'email':cr.email,
            'password':cr.password,
            'cpassword':cr.cpassword,
            'gender':cr.gender,
            }
        return render(request,'profile.html',user_info)
    else:
        return render(request,'profile.html')

def proupdate(request):
    email=request.session['email']
    if request.method == "POST":
        uname = request.POST.get('uname')
        email = request.POST.get('email')
        password=request.POST.get('password') 
        cpassword=request.POST.get('cpassword') 
        name=request.POST.get('name') 
        phone=request.POST.get('phone') 
        age=request.POST.get('age') 
        gender=request.POST.get('gender')
        dt=reg.objects.get(email=email)
        dt.name=name
        dt.uname=uname
        dt.email=email
        dt.password=password
        dt.cpassword=cpassword
        dt.phone=phone
        dt.age=age
        dt.gender=gender
        dt.save()
        response=redirect('/profile/')
        return response
        # return render(request,'profile.html')
    else:
        return render(request,'profile.html')



import cv2
import base64
from .models import normal  # Assuming 'normal' is your model for storing uploaded images

def checkdisease(request):
    if request.method == 'POST':
        img1 = request.FILES.get('img1')
        # Save the uploaded image
        normal_img = normal(img=img1)
        normal_img.save()

        # Read the uploaded image
        image_path = normal_img.img.path
        img = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Thresholding to identify black regions
        ret, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # Invert the binary image
        thresh = cv2.bitwise_not(thresh)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Flag variable to track if contours were found
        contours_found = False

        # Draw bounding boxes around black spots
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            contours_found = True

        # If no contours found, display the original uploaded image
        if not contours_found:
            _, buffer = cv2.imencode('.jpg', img)
            segmented_img = base64.b64encode(buffer).decode('utf-8')
            result_message = "no caries found"
        else:
            # Convert the image to base64 string for displaying in HTML
            _, buffer = cv2.imencode('.jpg', img)
            segmented_img = base64.b64encode(buffer).decode('utf-8')
            result_message = "caries exists"

        # Pass the processed image and result message to the HTML page to display
        return render(request, 'result.html', {'image': segmented_img, 'result_message': result_message})

    return render(request, 'fileupload.html')




from django.http import HttpResponse

from django.http import HttpResponse

import cv2
import base64
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
from .models import xray  # Assuming XRay is your Django model for X-ray images

def xrayupload(request):
    if request.method == 'POST':
        xrayimg = request.FILES.get('xrayimg')
        
        # Save the uploaded image to the XRay model
        xray1 = xray(ximg=xrayimg)
        xray1.save()

        # Get the path to the saved image
        xray_path = xray1.ximg.path
        
        # Ensure xray_path is valid
        if xray_path:
            # Read the image using OpenCV
            img = cv2.imread(xray_path)

            # Convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Threshold the grayscale image
            ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

            # Calculate area of white regions (damage areas) in the thresholded image
            area = np.sum(thresh == 255)
            damage_score = area / (img.shape[0] * img.shape[1])  # Normalize area by image size

            # Draw bounding boxes on detected areas (whole image or specific regions)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for idx, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                if idx % 2 == 0:
                    # Draw red bounding box for even index contours
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    # Draw green bounding box for odd index contours
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Encode the image with bounding boxes to base64 for rendering in HTML
            _, buffer = cv2.imencode('.jpg', img)
            segmented_img = base64.b64encode(buffer).decode('utf-8')

            # Render the segmented image and other data in HTML template
            return render(request, 'segmented_xray.html', {'segmented_img': segmented_img, 'damage_score': damage_score})

        else:
            return HttpResponse("Error: Segmentation failed. Could not find image path.")

    else:
        return render(request, 'xray.html')
def load_unet():
    # Load and preprocess the image
    predictions=''
    # Make predictions
    try:
        predictions = model.predict(image_path)
    except Exception as e:

        pass

    return predictions