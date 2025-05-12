# Phân Loại rác - ResNet50
# Tổng quan 
 Dự án này tập trung xây dựng mô hình Convolutional Neural Network (CNN) để phân loại hình ảnh rác thải nhựa thành nhiều loại khác nhau. Mục tiêu chính là tăng cường hệ thống quản lý chất thải bằng cách cải thiện quy trình phân loại và tái chế bằng công nghệ học sâu.
## Mục lục
- [Mô tả dự án](#project-description)  
- [Tập dữ liệu](#dataset)  
- [Kiến trúc mô hình](#model-architecture)  
- [Triển khai mô hình](#model-deployment)  
- [Đào tạo](#đào tạo)  
- [Tiến độ hàng tuần](#weekly-progress)  
- [Cách chạy](#cách chạy)  
- [Công nghệ được sử dụng](#technology-used)  
- [Phạm vi tương lai](#phạm vi tương lai)  
- [Đóng góp](#contributing)  
- [Giấy phép](#giấy phép)

## Mô tả dự án  
Ô nhiễm nhựa đang là mối lo ngại ngày càng tăng trên toàn cầu và việc phân loại rác thải hiệu quả là rất quan trọng để giải quyết vấn đề này. Dự án này sử dụng mô hình CNN để phân loại rác thải nhựa thành các loại riêng biệt, tạo điều kiện thuận lợi cho việc quản lý rác thải tự động.
## Bộ dữ liệu  
Tập dữ liệu được sử dụng cho dự án này là **Dữ liệu phân loại rác** .  Nó chứa tổng cộng 25.077 hình ảnh được gắn nhãn, được chia thành 3  loại: **Nhóm Tái Chế ** ,  **Rác thực phẩm** và **Các Loại Rác Khác**

### Chi tiết chính:
- **Tổng số hình ảnh**: 25.077  
  - **Dữ liệu huấn luyện**: 22.564 hình ảnh (85%)  
  - **Dữ liệu thử nghiệm**: 2.513 hình ảnh (15%)  
- **Lớp**:  Nhóm tái Chế , Rác Thực Phẩm , Các Loại Rác Khác   
- **Mục đích**: Để hỗ trợ tự động hóa việc quản lý chất thải và giảm tác động đến môi trường do việc xử lý chất thải không đúng cách.

### Dataset Link:  
Bạn có thể truy cập tập dữ liệu tại đây: [Dữ liệu phân loại rác thải](https://www.kaggle.com/datasets/techsash/waste-classification-data).  

## Kiến trúc mô hình  
Kiến trúc CNN bao gồm:  
- **Lớp chập:** Trích xuất tính năng  
- **Các lớp tổng hợp:** Giảm kích thước  
- **Các lớp được kết nối đầy đủ:** Phân loại  
- **Chức năng kích hoạt:** ReLU và Softmax

### Cấu trúc mô hình:
<p align="center">
  <img src="./anh.png/Kientruc_CNN.png" style="width:80%;">
</p>

## Triển khai mô hình  
Mô hình CNN được đào tạo có sẵn trên Kaggle:

[Waste Classification CNN Model](https://www.kaggle.com/models/hardikksankhla/waste-classification-cnn-model/)

## Đào tạo  
- **Trình tối ưu hóa:** Adam  
- **Hàm mất:** Entropy chéo phân loại  
- **Kỷ nguyên:** 25  
- **Kích thước lô:** 32

- **Các hoạt động:**  
  - Nhập các thư viện và framework cần thiết.  
  - Thiết lập môi trường dự án.  
  - Khám phá cấu trúc tập dữ liệu.
  
- **Notebooks:**  
  - [phanloairac.ipynb](phanloairac.ipynb)  
  - [Kaggle Notebook](https://www.kaggle.com/code/hardikksankhla/cnn-plastic-waste-classification)  

- **Notebooks:**  
  - [phanloairac.ipynb](phanloairac.ipynb)  
  - [Kaggle Notebook](https://www.kaggle.com/code/hardikksankhla/cnn-plastic-waste-classification)  


## Technologies Used  
- Python  
- TensorFlow/Keras  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  
- Streamlit  



# Quy trình thực hiện
## Bước 1: Chuẩn bị dữ liệu đầu vào
  Tiến hành thu thập và tổng hợp dữ liệu phân loại rác từ nhiều nguồn khác nhau. Sau đó, dữ liệu được lưu vào file CSV để dễ dàng xử lý và quan sát dưới dạng bảng, phục vụ cho các bước tiền xử lý và huấn luyện mô hình sau này.
## Bước 2: Chuyển đổi dữ liệu từ CSV sang PKL
  Việc chuyển dữ liệu từ định dạng CSV sang định dạng PKL (pickle) mang lại nhiều lợi ích thiết thực. Mặc dù CSV là định dạng phổ biến và dễ chia sẻ, nhưng nó còn hạn chế về hiệu suất và không thể giữ nguyên kiểu dữ liệu gốc. Ngược lại, định dạng PKL cho phép lưu trữ dữ liệu ở dạng nhị phân, giúp tăng tốc độ tải và đảm bảo bảo toàn các kiểu dữ liệu như chuỗi thời gian (datetime), kiểu phân loại (category) hay kiểu số (numeric). Nhờ đó, quá trình xử lý và huấn luyện mô hình trở nên nhanh chóng, hiệu quả và đáng tin cậy hơn trong môi trường Python.
## Bước 3: Huấn luyện mô hình phân loại rác
  Tiến hành chạy file phanloairac.ipynb để thực hiện việc huấn luyện mô hình phân loại rác. Sau khi hoàn tất, mô hình sẽ được lưu lại dưới dạng file Phanloairac_model.h5, đây là file mô hình đã được huấn luyện sẵn để sẵn sàng tích hợp vào ứng dụng thực tế thông qua giao diện web sử dụng Streamlit.

## Bước 4: Triển khai ứng dụng giao diện với Streamlit
  Sau khi có mô hình huấn luyện, tiến hành chạy ứng dụng giao diện bằng cách sử dụng file rac_1.py. Để chạy ứng dụng, sử dụng lệnh sau trong terminal:

  streamlit run rac_1.py
  Ứng dụng sẽ khởi chạy trên trình duyệt, cho phép người dùng tải ảnh lên và nhận kết quả phân loại rác trực tiếp từ mô hình đã huấn luyện.
