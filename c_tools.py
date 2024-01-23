from langchain.tools import tool
from PIL import Image
import base64
from io import BytesIO
import json
import cv2
import face_recognition
import numpy as np

known_face_encodings = []
known_face_names = []
face_locations = []
face_names = []
face_encodings = []
process_this_frame = True
def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format=pil_image.format)  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@tool
def nothing(query: str) -> str:
    """This tool is used when you don't have any action input. Be careful while using this tool and don't overuse it. THIS IS NOT THE USER'S ANSWER"""
    return f"Return the Final Answer. You said {query}."

@tool
def image_information_search(instruction: str) -> str:
    """This tool can be used to search for reliable information about a single image or sometimes faces. you pass the path of the file and the instructions and you want to figure out from it. You will give the input string in JSON format only with path and query as two possible value keys in the JSON string. You can use this reliable source for learning about images."""
    print("Image Recognition Tool INVOKED")
    try:
        js = json.loads(instruction)
    except:
        return "Invalid JSON format. Please give the input in JSON format only."
    
    try:
        file_path = js["path"]
        pil_image = Image.open(file_path)
    except:
        return "Invalid path. Give the correct path in the correct JSON format string or check if the given file exists or not."
    try:
        query = js["query"]
    except:
        return "Query not found. Give the correct query or check the JSON format input key."
    image_b64 = convert_to_base64(pil_image)

    from langchain_community.llms import Ollama
    bakllava = Ollama(model="bakllava")

    llm_with_image_context = bakllava.bind(images=[image_b64])
    
    out = llm_with_image_context.invoke(query)
    return out

@tool
def facial_recognition(path: str) -> str:
    """This tool can be used to recognise faces of the people asked. The name of the file is not necessarily the name of the person. Give the path to the image as a string in JSON FORMAT ONLY. USE THIS FOR HUMAN FACES ONLY."""
    print("Facial Recognition Tool INVOKED")
    print(known_face_encodings)
    print(known_face_names)
    if not known_face_encodings:
        return "No known faces to recognise. Try searching for the face in a reliable source and then add the face."
    if len(known_face_encodings) != len(known_face_names):
        return "Known face encodings and names are not equal. Please check the code"
    process_this_frame = True
    try:
        js = json.loads(path)
    except:
        return "Invalid JSON format. Please give the input in JSON format only."
    try:
        frame = face_recognition.load_image_file(js["path"])
        print(type(frame))

    except:
        return "Invalid path. Give the correct path in the correct JSON format string or check if the given file exists or not."
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            print(best_match_index)
            if matches[best_match_index-1]:
                name = known_face_names[best_match_index-1]
                face_names.append(name)
    process_this_frame = not process_this_frame
    if not face_names:
        return f"No faces recognised. it is no one in list of \"{' '.join(names for names in known_face_names)}\". You can use the image search tool to get information about this face and then add the new face using the appropriate tool."
    return "The face is of " + "\n".join(names for names in face_names)
    
@tool
def new_face_addition(info: str) -> str:
    """This tool can be used to add new faces to the known face encodings after recognising an unknown face by any means. You will recieve the path of the image and the name of the person as a string JSON format only. The two variables will be path and name. USE THIS FOR HUMAN FACES ONLY."""
    print("New Face Addition Tool INVOKED")
    try:
        js = json.loads(info)
        name = js["name"]
    except:
        return "Invalid JSON format. Please give the input in JSON format only or check if the file exists or not."
    
    try:
        face_encoding = face_recognition.face_encodings(face_recognition.load_image_file(js["path"]))[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
    except:
        return "Invalid JSON format. Please give the input in JSON format only."
    return f"Face with the name {name} Added Successfully."


@tool
def add_important_context(info: str) -> str:
    """This tool can be used to add context or details from the conversation. You will give the context as a string in JSON format only. The variable will be context. You can use this context in the future and build up an understanding about your master."""
    print("Important Context Addition Tool INVOKED")
    try:
        js = json.loads(info)
        context = js["context"]
    except:
        return "Invalid JSON format. Please give the input in JSON format only."
    
    with open("context.txt", "a") as f:
        f.write(context + "\n")
    return f"Context Added Successfully. Context: {context}"