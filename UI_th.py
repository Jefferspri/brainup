# Third party moduls
from pygame import mixer
import _tkinter
import tkinter as tk
from tkinter import ttk
import datetime
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageSequence
import matplotlib.pyplot as plt
from scipy import stats
import random
from random import shuffle
from typing import List, Tuple
import sys
import threading
from worker import create_worker,listen, sleep
import asyncio
import nest_asyncio
nest_asyncio.apply()
from async_tkinter_loop import async_handler, async_mainloop
from pylsl import StreamInlet, resolve_byprop
import math
from scipy import signal
from scipy.signal import filtfilt
import spkit as sp
import pywt
from sklearn import metrics
import pickle
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from scipy.ndimage import gaussian_filter1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

# own moduls
from moduls import tr_moduls as trm
from moduls.neurofeedback import record
from moduls import process_functions as pfunc

import warnings
warnings.filterwarnings("ignore")


class FirstPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(bg='#fafafa')
        
        label_desc = tk.Label(self, text="""\n\n
                             Esta es una prueba de atención, prueba gradCPT.
                             En ella observarás como pasan imágenes de forma gradual,
                             imágenes de ciudades y montañas. Cada vez que veas la 
                             imagen de una ciudad, debes presionar el botón azul.
                                             
                             Presiona continuar para ver las imágenes que encontraras
                             en la prueba. Luego, "Entrenar" para acostumbrarse a la 
                             prueba. Finalmente, "Iniciar"" para hacer la prueba completa.""",
                             fg="#4d4e4f", bg='#fafafa', font=("Arial", 14))
        label_desc.place(x=2, y=5)
        
        Button = tk.Button(self, text="Continuar", font=("Arial", 14), command=lambda: controller.show_frame(SecondPage))
        Button.place(x=590, y=300)
        
        self.bind('<Enter>', self.enter)
        
    def enter(self, event):
        if mixer.music.get_busy():
            mixer.music.stop()
        
        

class SecondPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        self.configure(bg='#fafafa')
        
        self.img_num = 1
        imagia = Image.open("pics/images/c1.jpg").convert('L')
        imagia = imagia.resize((350, 350), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(imagia)
        
        self.lbl_img = tk.Label(self, image=img)
        self.lbl_img.image = img 
        self.lbl_img.place(x=140,y=25)
        
        
        Button = tk.Button(self, text="Atrás", font=("Arial", 14), command=lambda: controller.show_frame(FirstPage))
        Button.place(x=612, y=7)

        Button = tk.Button(self, text="<", font=("Arial", 14), command=self.back_img)
        Button.place(x=600, y=100)
        Button = tk.Button(self, text=">", font=("Arial", 14), command=self.next_img)
        Button.place(x=650, y=100)

        Button = tk.Button(self, text="Entrenar", font=("Arial", 14), command=lambda: controller.show_frame(ThirdPage))
        Button.place(x=600, y=200)
        
        Button = tk.Button(self, text="Iniciar", font=("Arial", 14), command=lambda: controller.show_frame(FourthPage))
        Button.place(x=610, y=300)
        
        self.bind('<Enter>', self.enter)
        
    def enter(self, event):
        if mixer.music.get_busy():
            mixer.music.stop()
    
        
    def next_img(self):
        if self.img_num < 20:
            self.img_num += 1
            imagia = Image.open(f"pics/images/c{self.img_num}.jpg").convert('L')
            imagia = imagia.resize((350, 350), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(imagia)
            self.lbl_img.config(image=img)
            self.lbl_img.image = img 
            self.update()
       
    def back_img(self):
        if self.img_num > 1:
            self.img_num -= 1
            imagia = Image.open(f"pics/images/c{self.img_num}.jpg").convert('L')
            imagia = imagia.resize((350, 350), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(imagia)
            self.lbl_img.config(image=img)
            self.lbl_img.image = img
            self.update()
        
        

class ThirdPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(bg='#fafafa')
        
        self.lbl_img = tk.Label(self)
        self.lbl_img.place(x=140,y=25)

        Button = tk.Button(self, text="Atrás", font=("Arial", 14), command=lambda: controller.show_frame(SecondPage))
        Button.place(x=605, y=7)
        
        Button = tk.Button(self, text="Iniciar", font=("Arial", 14), command=lambda: controller.show_frame(FourthPage))
        Button.place(x=605, y=70)
        
        Button = tk.Button(self, text="O", font=("Arial", 14), command = async_handler(self.play_gif))
        Button.place(x=620, y=150)
        
        Button = tk.Button(self, text="            ", font=("Arial", 14), bg="#036ffc")
        Button.place(x=600, y=300)
        
    def creador_de_lista_final(self, tam):
        lista_final=[]
        lista=[]
        for j in range(tam):#el 60 es para 8 minutos, 6 es para prueba de 48 seg, 15 para 2 minutos
            while(len(lista)<9): # inserta las ciudades
                x=random.randint(1,10)
                x=str(x)
                y='pics/grays/ct{}.jpg'.format(x)
                
                if y not in lista:
                    lista.append(y)
            # inserta una montana en una posicion random
            lista.insert(random.randrange(1,10),'pics/grays/mt{}.jpg'.format(random.randrange(1,11)))
            lista_final = lista_final + lista
            lista = []
        return lista_final
        
        
    async def play_gif(self):
        images = []
        lst_random_images = self.creador_de_lista_final(15)
        
        len_list_images = len(lst_random_images)
        for l in range (len_list_images - 1):
            img_act = lst_random_images[l]
            img_sig = lst_random_images[l+1]

            with Image.open(img_act) as source_img, Image.open(img_sig) as dest_img:
                # add start image
                images.append(ImageTk.PhotoImage(source_img))
                
                for tran in range(3):
                    source_img = Image.blend(source_img, dest_img, 0.25*(tran+1))
                    images.append(ImageTk.PhotoImage(source_img))
                    await asyncio.sleep(0.0001)
        
        mixer.music.play(0)
        print("Test Started.")
        
        for img in images:
            #img = ImageTk.PhotoImage(i)
            self.lbl_img.config(image=img)
            self.update()
            await asyncio.sleep(0.2)
            if mixer.music.get_busy() == False:
                break
        
        mixer.music.stop()
        
        print("Test Stopped.")
            
        
        
class FourthPage(tk.Frame):

    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        self.configure(bg='#fafafa')
        
        self.pb = ttk.Progressbar(self, orient='horizontal',
                                    mode='determinate',
                                    length=350)
        self.pb.place(x=141,y=3)
        self.pb.step(0)
                
        self.lbl_img = tk.Label(self)
        self.lbl_img.place(x=140,y=25)

        btn_home = tk.Button(self, text="Home", font=("Arial", 14), command=lambda: controller.show_frame(FirstPage))
        btn_home.place(x=605, y=7)

        btn_back = tk.Button(self, text="Atrás", font=("Arial", 14), command=lambda: controller.show_frame(SecondPage))
        btn_back.place(x=607, y=50)      
        
        self.btn_start = tk.Button(self, text="O", font=("Arial", 14), command = async_handler(self.play_gif)) #lambda:[self.play_gif, record]) # record 
        self.btn_start.place(x=620, y=150)
        
        self.btn_result = tk.Button(self, text="Resultados", font=("Arial", 12), command=lambda: controller.show_frame(FivePage))
        self.btn_result.place(x=590, y=200)
        self.btn_result["state"] = "disabled"
        
        btn_click = tk.Button(self, text="            ", font=("Arial", 14), bg="#036ffc", command=self.take_time)
        btn_click.place(x=600, y=300)
        
        self.n = 1
        self.last = 0
        
        self.bind('<Enter>', self.enter)
        
    def enter(self, event):
        if mixer.music.get_busy() and recording==False:
            mixer.music.stop()
            
    def take_time(self):
        global t_details 
        t_details["time"].append(datetime.datetime.now()) # transition start time
        t_details["tag"].append("click")

    def creador_de_lista_final_8(self, tam):
        lista_final=[]
        lista=[]
        for j in range(60):#el 60 es para 8 minutos, 6 es para prueba de 48 seg, 15 para 2 minutos
            while(len(lista)<9): # inserta las ciudades
                x=random.randint(1,10)
                x=str(x)
                y='pics/grays/ct{}.jpg'.format(x)
                
                if y not in lista:
                    lista.append(y)
            # inserta una montana en una posicion random
            lista.insert(random.randrange(1,10),'pics/grays/mt{}.jpg'.format(random.randrange(1,11)))
            lista_final = lista_final + lista
            lista = []
        return lista_final
        

    async def play_gif(self): 
        global recording
        global auxCount
        global t_details
        global all_raw_data
        t_details = {"time":[], "tag":[], "tr":[], "flag":[]}
        all_raw_data = {'eeg':[], 'time':[]}
        images=[]
        self.pb["value"] = 0 # progress bar value
        
        lst_random_images = self.creador_de_lista_final_8(60)
        
        #imagia = Image.open("prueba-8-min.gif")
        
        self.btn_start["state"] = "disabled"
        
        len_list_images = len(lst_random_images)
        for l in range (len_list_images - 1):
            img_act = lst_random_images[l]
            img_sig = lst_random_images[l+1]
            
            self.pb["value"] = (int(100*(l+2)/len_list_images)-0.1) 

            with Image.open(img_act) as source_img, Image.open(img_sig) as dest_img:
                # add start image
                images.append(ImageTk.PhotoImage(source_img))
                
                for tran in range(3):
                    source_img = Image.blend(source_img, dest_img, 0.25*(tran+1))
                    images.append(ImageTk.PhotoImage(source_img))
                    await asyncio.sleep(0.0001)  
        
        # Save the images as a GIF - asegurate de guardar el formato correcto de imagen
        #images[0].save("gifs/p3-8-min.gif", save_all=True, append_images=images[1:], duration=200, loop=1)
    
        
        recording = True
        mixer.music.play(0)
        self.btn_result["state"] = "normal"
        print("Recording Started.")
        
        #inicio = time.time()
        counter = 1
        for i in range(len(images)):
            self.lbl_img.config(image = images[i])
            self.update()
            if i%4 == 0:
                #print(time.time()- inicio)
                #inicio = time.time()
                t_details["time"].append(datetime.datetime.now()) # transition start time
                t_details["tag"].append(lst_random_images[counter])
                counter += 1
                if mixer.music.get_busy() == False:
                    recording = False
                    break
                
            await asyncio.sleep(0.19)
       
        
        # Closing process
        recording = False
        mixer.music.stop()
        #f.close()
        self.btn_start["state"] = "normal"
        
        print("Recording Stopped.")
   


class FivePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(bg='#fafafa')
        
        self.lbl_tr = tk.Label(self, text="TR promedio: ", bg='#fafafa', font=("Arial", 12))
        self.lbl_tr.place(x=2, y=5)
        self.lbl_correct = tk.Label(self, text="Correctas: ", bg='#fafafa', font=("Arial", 12))
        self.lbl_correct.place(x=2, y=27)
        self.lbl_lapse = tk.Label(self, text="Errores: ", bg='#fafafa', font=("Arial", 12))
        self.lbl_lapse.place(x=2, y=49)
        self.lbl_tr_modelo = tk.Label(self, text="TR promedio modelo: ", bg='#fafafa',fg='#203ee6' ,font=("Arial", 12))
        self.lbl_tr_modelo.place(x=2, y=82)
        self.lbl_mae = tk.Label(self, text="MAE: ", bg='#fafafa',fg='#203ee6' , font=("Arial", 12))
        self.lbl_mae.place(x=2, y=104)
        self.lbl_p_error = tk.Label(self, text="Error: ", bg='#fafafa',fg='#203ee6' , font=("Arial", 12))
        self.lbl_p_error.place(x=2, y=126)

        btn_home = tk.Button(self, text="Home", font=("Arial", 14), command=lambda: controller.show_frame(FirstPage))
        btn_home.place(x=350, y=5)
        
        self.btn_result = tk.Button(self, text="Resultados", font=("Arial", 12), command=self.show_tr)
        self.btn_result.place(x=350, y=45)
        
        frame1 = tk.Frame(self, width=300, height=200, background="bisque")
        frame1.place(x=12, y=150) 

        frame2 = tk.Frame(self, width=200, height=700, background="bisque")
        frame2.place(x=470, y=0)

        # create a figure
        figure = Figure(figsize=(4.5, 2.5), dpi=100)
        figure2 = Figure(figsize=(3.5, 3.75), dpi=100)

        # create FigureCanvasTkAgg object
        self.figure_canvas = FigureCanvasTkAgg(figure, frame1)
        self.figure_canvas2 = FigureCanvasTkAgg(figure2, frame2)

        # create the toolbar
        NavigationToolbar2Tk(self.figure_canvas2, frame2)

        # create axes
        self.axes = figure.add_subplot()
        self.figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # create axes
        self.axes2 = figure2.add_subplot()
        self.figure_canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        #figure_canvas2.show()
        
        self.bind('<Enter>', self.enter)
        
    def enter(self, event):
        global recording
        if mixer.music.get_busy():
            self.lbl_tr.configure(text="TR promedio: ") 
            self.lbl_correct.configure(text="Correctas: ") 
            self.lbl_lapse.configure(text="Errores: ")
            recording = False
            mixer.music.stop()
            #f.close()
        
    def show_tr(self):
        global t_details
        global all_raw_data

        t_details["tr"].append(float('nan'))
        t_details["flag"].append("")

        details = trm.clean_trs(t_details)
        df_details = pd.DataFrame.from_dict(details)
        df_details.to_csv(fileTimes, index=False)
        
        # mean TR # this TR no consider the range 0.56 to 1.12
        lst_re_times = [i for i in details["tr"] if not math.isnan(i)]
        tr_mean = np.mean(lst_re_times)
        # correct and lapse
        lst_correct = ["correct" for i in details["flag"] if i == "correct comission" or i == "correct omission"]
        lst_lapse = ["lapse" for i in details["flag"] if i == "comission error" or i=="omission error"]
        # VTC
        tr_mean_vtc = np.mean(lst_re_times)
        tr_var_vtc = np.std(lst_re_times)
        # values for error trials (CEs and OEs) and
        # correct omissions (COs) were interpolated linearly—that is,
        # by weighting the two neighboring baseline trial RTs.
    
        VTC= [abs(i - tr_mean_vtc)/tr_var_vtc  for i in lst_re_times]
        
        
        # EEG processing raw  - 'TP9', 'AF7', 'AF8', 'TP10', 'Right AUX'
        all_raw_data['TP9'] = []
        all_raw_data['AF7'] = []
        all_raw_data['AF8'] = []
        all_raw_data['TP10'] = []
        all_raw_data['Right AUX'] = []
        
        for raw in all_raw_data['eeg']:
            all_raw_data['TP9'].append(raw[0])
            all_raw_data['AF7'].append(raw[1])
            all_raw_data['AF8'].append(raw[2])
            all_raw_data['TP10'].append(raw[3])
            all_raw_data['Right AUX'].append(raw[4])
            
        all_raw_data.pop('eeg') # drop column with data now separte in various columns
        df_raw = pd.DataFrame.from_dict(all_raw_data)
        df_raw.to_csv(filePath, index=False)
        
        # Formating Data
            # df_raw  : eeg data
            # df_details : click times
        df_eeg, df_rt = pfunc.formating_data(df_raw, df_details)
        
        # Generate df rt range date
        df_rt_date = pfunc.generate_df_rt_date_no_mean(df_rt)
        
        # Preprocessing EEG data
        df_eeg = pfunc.preprocessimg_data(df_eeg)
        
        # Wavelet decomposition - Characteristics extraction
        df_features = pfunc.wavelet_packet_decomposition(df_eeg, df_rt_date)
        
        # Date normalization
        df_features = pfunc.normalization(df_features)
        
        # Pivot all channel characteristics to columns
        df_all_features = pfunc.pivot_channels(df_features)
        
        # Import SVR model and predict
        regressor = pickle.load(open('models/svr_model.sav', 'rb'))  
        
        X_test = df_all_features.iloc[:,:df_all_features.shape[1]-4].values
        y_test = df_all_features.iloc[:,df_all_features.shape[1]-4].values
        
        y_pred = regressor.predict(X_test)
        x = [i for i in range(len(y_pred))]
        
        # show standard calculation
        self.lbl_tr.configure(text="TR promedio: {:.4f} seg".format(np.mean(y_test)))
        self.lbl_correct.configure(text=f"Correctas: {len(lst_correct)}") 
        self.lbl_lapse.configure(text=f"Errores: {len(lst_lapse)}")  
        
        self.lbl_tr_modelo.configure(text="TR promedio modelo: {:.4f} seg".format(np.mean(y_pred)))
        self.lbl_mae.configure(text="MAE: {:.4f} seg".format(metrics.mean_absolute_error(y_test, y_pred)))
        self.lbl_p_error.configure(text="Error: {:.2f}%".format(np.mean(100*abs(y_pred-y_test)/y_test)))
        
        # Plot
        self.axes.clear()
        major_locator =FixedLocator(x)
        self.axes.xaxis.set_major_locator(major_locator)
        self.axes.scatter(x[:21] ,y_test[:21]  , color= '#07D99A') # estándar
        self.axes.scatter(x[:21]  ,y_pred[:21]  , color='#203ee6', marker="x") # modelo
        self.axes.set_title('Tiempos de Respuesta', fontsize=10)
        self.axes.set_ylabel('TR (seg.)')
        self.axes.set_xlabel('Muestra')
        self.axes.legend(['real', 'modelo'])
        self.axes.grid(axis='x')
        self.figure_canvas.draw()
        
        # Plot all click and power
        self.axes2.clear()
      
        data = df_features[df_features['channel']=='AF8_fil']
        data['cont'] = np.linspace(0,8,data.shape[0])

        mask_correct = (data['flag']== 'correct comission')|(data['flag']== 'correct omission')
        mask_error = (data['flag']== 'comission error')|(data['flag']== 'omission error')

        t_correct = data['cont'][mask_correct]
        point_correct = [0.6 for i in t_correct]
        t_error = data['cont'][mask_error]
        point_error = [0.5 for i in t_error]

        data = data.dropna()
        x = data['cont'].to_list()
        tr = data['tr'].to_list()
        tr_smooth = gaussian_filter1d(tr, sigma=2)

        power = gaussian_filter1d(data['p_beta']+0.5, sigma=2)

        # create the barchart
        self.axes2.plot(x, tr_smooth, color= '#a0a0a3') # estándar
        self.axes2.plot(x, power, color='#203ee6') # modelo
        self.axes2.scatter(t_error, point_error, color= '#e8053d') # estándar
        self.axes2.scatter(t_correct, point_correct, color='#1de096') # modelo
        self.axes2.set_title('TR y Potencia en el tiempo', fontsize=10)
        self.axes2.legend(['tr', 'potencia', 'error', 'correcto'])
        self.axes2.grid(True)
        self.figure_canvas2.draw()
        
        

class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # creating a window
        window = tk.Frame(self)
        window.pack()

        window.grid_rowconfigure(0, minsize=500)
        window.grid_columnconfigure(0, minsize=800)

        self.frames = {}
        for F in (FirstPage, SecondPage, ThirdPage, FourthPage, FivePage):
            frame = F(window, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(FirstPage)
        
        
    def show_frame(self, page):
        frame = self.frames[page]
        frame.tkraise()
        self.title("Prueba de atención")
    

def eeg_writer():
    global all_raw_data
    global recording
    #global end
    # This buffer will hold last n seconds of data and be used for calculations
    BUFFER_LENGTH = 5
    # Length of the epochs used to compute the FFT (in seconds)
    EPOCH_LENGTH = 1
    # Amount of overlap between two consecutive epochs (in seconds)
    OVERLAP_LENGTH = 0.8
    # Amount to 'shift' the start of each next consecutive epoch
    SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
        
    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()
    
    # Get the stream info and description
    info = inlet.info()
    description = info.desc()
    
    fs = int(info.nominal_srate())
    # capture
    while True:
        #
        try:
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            if recording:
                eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(SHIFT_LENGTH * fs))
                print(eeg_data[0])
                all_raw_data['eeg'] = all_raw_data['eeg'] + eeg_data
                all_raw_data['time'] = all_raw_data['time'] + timestamp
            else:
                time.sleep(2)
        #    """     
        except IndexError:
            # Sleep 10 sec.
            time.sleep(5)
            
            print('Looking for an EEG stream...')
            streams = resolve_byprop('type', 'EEG', timeout=2)
            print(streams)
            if len(streams) == 0:
                print('Can\'t find EEG stream.')
            else:
                # Set active EEG stream to inlet and apply time correction
                print("Start acquiring data")
                inlet = StreamInlet(streams[0], max_chunklen=12)
                info = inlet.info()
                fs = int(info.nominal_srate())
                     


def on_closing():
    global end
    global app
    end = True
    app.destroy()
    
    
def task_tk():
    #global app
    app = App()
    #app.protocol("WM_DELETE_WINDOW", on_closing)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_mainloop(app))
       
     
# a little necesary time to don't interrupt eeg capture init         
time.sleep(1)

# Initialize variables
recording = False # EEG recording status
end = False
filePath = "exports/eeg_data.csv"
fileTimes = "exports/times_data.csv"
all_raw_data = {'eeg':[], 'time':[]}

# Initialize music
mixer.init(44100)
mixer.music.load("lofi-mod.mp3")
mixer.music.play(0)
mixer.music.stop()

# Initialize threading and azync process
try:
    th_eeg = threading.Thread(target = eeg_writer)
    th_tk = threading.Thread(target = task_tk)
    th_tk.start()
    th_eeg.start()
except:
    pass
finally:
    print("Closing")
    mixer.music.stop()
