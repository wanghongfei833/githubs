import os
import tkinter as tk
from tkinter.filedialog import *

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class UI(object):
	def __init__(self):
		super(UI, self).__init__()
		self.root = tk.Tk()
		self.imageListShow = []
		self.file = tk.StringVar()
		self.fig = plt.figure()
		self.canv = FigureCanvasTkAgg(self.fig, master=self.root)
		self.canv.draw()
		self.canv._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
	
	def files(self):
		self.file.set(askdirectory())
		self.imageListShow = os.listdir(self.file.get())
		self.imageListShow = [os.path.join(self.file.get(), i) for i in self.imageListShow]
		self.imageShowIndex = 0
	
	def prints(self):
		print(self.imageShowIndex)
		self.root.after(1000, self.prints)
	
	def Farm(self):
		tk.Button(self.root, text='load', command=self.files).pack()
		tk.Button(self.root, text='show', command=self.prints).pack()
	
	def show(self):
		self.fig.clf()
		img = plt.imread(self.imageListShow[self.imageShowIndex])
		self.imageShowIndex += 1
		g11 = self.fig.add_axes([0.1, 0.1, 0.85, 0.85])
		g11.imshow(img)
		self.canv.draw()
		if self.imageShowIndex < len(self.imageListShow): self.root.after(1000, self.show)
	
	def run(self):
		self.Farm()
		self.show()
		self.root.mainloop()


if __name__ == '__main__':
	mains = UI()
	mains.run()