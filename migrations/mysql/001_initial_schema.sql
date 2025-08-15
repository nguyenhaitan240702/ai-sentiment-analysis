-- MySQL database initialization for AI Sentiment Analysis
-- Creates tables for analytics and reporting

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS aisent CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE aisent;

-- Daily sentiment statistics table
CREATE TABLE IF NOT EXISTS sentiments_daily (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    lang VARCHAR(5) NOT NULL DEFAULT 'vi',
    source VARCHAR(50) NOT NULL DEFAULT 'api',
    pos_cnt INT NOT NULL DEFAULT 0,
    neu_cnt INT NOT NULL DEFAULT 0,
    neg_cnt INT NOT NULL DEFAULT 0,
    total_cnt INT GENERATED ALWAYS AS (pos_cnt + neu_cnt + neg_cnt) STORED,
    avg_score DECIMAL(4,3) DEFAULT 0.000,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_daily_stats (date, lang, source),
    INDEX idx_date_lang (date, lang),
    INDEX idx_source (source),
    INDEX idx_last_updated (last_updated)
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL DEFAULT '1.0',
    date DATE NOT NULL,
    total_requests INT NOT NULL DEFAULT 0,
    avg_latency_ms DECIMAL(8,2) DEFAULT 0.00,
    cache_hit_rate DECIMAL(5,2) DEFAULT 0.00,
    error_rate DECIMAL(5,2) DEFAULT 0.00,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY unique_model_daily (model_name, model_version, date),
    INDEX idx_model_date (model_name, date)
);

-- API usage statistics
CREATE TABLE IF NOT EXISTS api_usage (
    id INT AUTO_INCREMENT PRIMARY KEY,
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INT NOT NULL,
    response_time_ms INT,
    user_agent TEXT,
    ip_address VARCHAR(45),
    request_date DATE NOT NULL,
    request_hour TINYINT NOT NULL,
    request_count INT NOT NULL DEFAULT 1,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY unique_usage_hour (endpoint, method, status_code, request_date, request_hour),
    INDEX idx_endpoint_date (endpoint, request_date),
    INDEX idx_status_date (status_code, request_date)
);

-- Language detection statistics
CREATE TABLE IF NOT EXISTS language_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detected_lang VARCHAR(5) NOT NULL,
    provided_lang VARCHAR(5),
    confidence_score DECIMAL(4,3),
    date DATE NOT NULL,
    count INT NOT NULL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY unique_lang_daily (detected_lang, provided_lang, date),
    INDEX idx_date_lang (date, detected_lang)
);

-- Insert initial data
INSERT IGNORE INTO sentiments_daily (date, lang, source, pos_cnt, neu_cnt, neg_cnt, avg_score)
VALUES
    (CURDATE(), 'vi', 'api', 0, 0, 0, 0.000),
    (CURDATE(), 'en', 'api', 0, 0, 0, 0.000);

-- Create user for the application
CREATE USER IF NOT EXISTS 'aisent_user'@'%' IDENTIFIED BY 'aisent_pass';
GRANT SELECT, INSERT, UPDATE, DELETE ON aisent.* TO 'aisent_user'@'%';
FLUSH PRIVILEGES;
