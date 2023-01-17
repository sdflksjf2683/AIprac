# 필요한 라이브러리 호출
import face_recognition
import os
# %matplotlib inline
# import matplotlib.pyplot as plt
import numpy as np

def get_cropped_face(image_file):

    image = face_recognition.load_image_file(image_file)    # 이미지 불러오기
    face_locations = face_recognition.face_locations(image)   # 얼굴 영역 박스 
    a, b, c, d = face_locations[0]     # 얼굴 영역 박스 좌표
    cropped_face = image[a:c,d:b,:]    # 얼굴 영역 박스 좌표를 이용해 얼굴 잘라내기 
    return cropped_face              
    
# image_path = './sh.jpg'

# 얼굴 잘라내는 함수 호출
# cropped_face = get_cropped_face(image_path)
# plt.imshow(cropped_face)    # 잘라낸 이미지 출력
# plt.show() # 잘라낸 이미지 확인

dir_path = './img/'
# file_list = ['round2', 'round3', 'round4', 'round5']
# file_list2 = 'sh.jpg'
# image_file = os.path.join(dir_path, file_list2)
# face = get_cropped_face(image_file)   # 얼굴 영역을 구하는 함수(이전 스텝에서 구현)

def get_face_embedding(face):
    return face_recognition.face_encodings(face)   # FaceNet 얼굴 임베딩 모델 이용

def get_face_embedding_dict(dir_path):
    file_list = os.listdir(dir_path)
    embedding_dict = {}
    
    for file in file_list:
        img_path = os.path.join(dir_path, file) # 경로를 병합하여 새 경로 생성
        try: 
            face = get_cropped_face(img_path)    # 얼굴 영역만 자른 이미지
        except:                                  # 인식하지 못하는 이미지는 넘어감
            continue
            
        embedding = get_face_embedding(face)   # 얼굴 영역에서 얼굴 임베딩 벡터를 추출
        if len(embedding) > 0:   # 얼굴 영역이 제대로 detect되지 않았을 경우를 대비
                    # os.path.splitext(file)[0]에는 이미지파일명에서 확장자를 제거한 이름이 담긴다. 
                embedding_dict[os.path.splitext(file)[0]] = embedding[0]
       
    return embedding_dict

embedding_dict = get_face_embedding_dict(dir_path)
def get_distance(name1, name2):
    return np.linalg.norm(embedding_dict[name1]-embedding_dict[name2], ord=2)

# print(get_distance('round2', 'round1'))

# name1과 name2의 거리를 비교하는 함수
def get_sort_key_func(name1):         # name1은 미리 지정
    def get_distance_from_name1(name2):      # name2는 호출시에 인자로 받는다.
        return get_distance(name1, name2)
    return get_distance_from_name1

# 유사한 이미지 찾는 함수
def get_nearest_face(name, top= len(os.listdir(dir_path))-1):
    sort_key_func = get_sort_key_func(name)  
    sorted_faces = sorted(embedding_dict.items(), key=lambda x:sort_key_func(x[0]))   # 얼굴 임베딩 딕셔너리를 오름차순으로 정렬
    
    for i in range(top+1):
          if i == 0:
              continue
          if sorted_faces[i]:
                print('{}: {}'.format(i,len(sorted_faces[i][0])))
                print('순위 {} : 이름({}), 거리({})'. format(i, sorted_faces[i][0], sort_key_func(sorted_faces[i][0])))

get_nearest_face('team7')