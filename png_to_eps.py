import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  
import cv2 as cv
def convert_png_to_eps(png_path, eps_path):  
    # 读取PNG图像  
    img = mpimg.imread(png_path)  
    print(img.shape)
    cv.imwrite(eps_path, img)
    return
    # 创建一个图形和轴对象  
    fig, ax = plt.subplots() 
    
    # 隐藏坐标轴  
    ax.axis('off')  
      
    # 显示图像  
    ax.imshow(img)  
      
    # 保存为EPS格式  
    plt.savefig(eps_path)  # dpi可以根据需要调整，以改善EPS图像的质量  
      
    # 关闭图形  
    plt.close(fig)  
  
# 使用示例  
png_path = "C:\\Users\\wispytrace\\Desktop\\figure\\blurred.png"  # 替换为你的PNG图像路径  
eps_path = 'H:\\code\\FXT\\temp\\blurred.eps'  # 输出的EPS文件路径  
convert_png_to_eps(png_path, eps_path)