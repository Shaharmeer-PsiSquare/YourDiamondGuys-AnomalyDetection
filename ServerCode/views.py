import json, codecs
import traceback

import numpy as np
# from django.shortcuts import render
from datetime import datetime
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from image_instance_segmentation import prediction
import os
import glob

# Create your views here.








class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@csrf_exempt
def index(request):
    global encodeimg, polygon_points, res_img, ori_path
    folder = 'static/input_img/'

    if request.method == "POST" :
        try:
            encodeflag = False
            file = request.FILES.get('file')
            incoded_img = request.POST.get('encode')
            if incoded_img is not None:
                encodeflag = True
                encodeimg, polygon_points, flag, center_falg,region_dict,accuracy_per, head_anomaly = prediction(None,encode_flag=encodeflag,encode_image=incoded_img)
            else:
                # url = request.get_host()
                input_img = glob.glob('static/input_img/*')
                for f in input_img:
                    os.remove(f)
                result = glob.glob('static/result/*')
                for f in result:
                    os.remove(f)
                anno = glob.glob('static/anno/*')
                for f in anno:
                    os.remove(f)
                location = FileSystemStorage(location=folder)
                fn = location.save(file.name, file)
                path = os.path.join('static/input_img/', fn)


                res_img, flag, ori_path, center_falg,region_dict,accuracy_per,head_anomaly = prediction(path,encode_flag=encodeflag,encode_image=None)
            if flag == True:
                if incoded_img is not None:
                    context = {
                        "status": flag,
                        "encode_img": f'{encodeimg}',
                        "polygon_points": polygon_points,
                        "center_flag": center_falg,
                        "region_points": json.dumps(region_dict, cls=NumpyEncoder),
                        "acuuracy": accuracy_per,
                        "anomaly_head":json.dumps(head_anomaly,cls=NumpyEncoder)
                    }
                else:
                    context = {
                        "status": flag,
                        "Image_path": f'{res_img}',
                        "orignal_path": ori_path,
                        "center_falg": center_falg,
                        "acuuracy": accuracy_per,
                        "region_points": json.dumps(region_dict, cls=NumpyEncoder),
                        "anomaly_head": json.dumps(head_anomaly, cls=NumpyEncoder)
                    }
            else:
                context = {
                    "status": "please send the valid image",
                }
            return JsonResponse(context)
        except Exception as _:
            message = traceback.format_exc()
            context = {
                "status": message,
            }
            return JsonResponse(context)

    return render(request, 'index.html')

