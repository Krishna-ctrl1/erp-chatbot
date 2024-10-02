CREATE DATABASE streamlit_app_db;
CREATE USER 'streamlit_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON streamlit_app_db.* TO 'streamlit_user'@'localhost';
FLUSH PRIVILEGES;

USE streamlit_app_db;
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL
);

