CREATE DATABASE IF NOT EXISTS `summer_project_2026`
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE `summer_project_2026`;

CREATE TABLE IF NOT EXISTS `cameras` (
  `id` VARCHAR(64) NOT NULL,
  `name` VARCHAR(255) NOT NULL,
  `url` VARCHAR(1024) NOT NULL,
  `longitude` DOUBLE NOT NULL,
  `latitude` DOUBLE NOT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `camera_stats` (
  `camera_id` VARCHAR(64) NOT NULL,
  `total_vehicle_count` INT NOT NULL DEFAULT 0,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`camera_id`),
  CONSTRAINT `fk_camera_stats_camera`
    FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `users` (
  `user_id` INT NOT NULL AUTO_INCREMENT,
  `user_name` VARCHAR(255) NOT NULL,
  `user_password` VARCHAR(255) NOT NULL,
  `user_type` VARCHAR(64) NOT NULL,
  `phone` VARCHAR(32) NULL,
  `personnel_category` VARCHAR(64) NOT NULL DEFAULT 'traffic_police',
  `site` VARCHAR(255) NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `uk_users_user_name` (`user_name`),
  INDEX `idx_users_user_type` (`user_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `admin_users` (
  `admin_user_id` INT NOT NULL AUTO_INCREMENT,
  `user_name` VARCHAR(255) NOT NULL,
  `user_password` VARCHAR(255) NOT NULL,
  `user_type` VARCHAR(64) NOT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`admin_user_id`),
  UNIQUE KEY `uk_admin_users_user_name` (`user_name`),
  INDEX `idx_admin_users_user_type` (`user_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `work_orders` (
  `work_order_id` INT NOT NULL AUTO_INCREMENT,
  `event_id` VARCHAR(64) NULL,
  `camera_id` VARCHAR(64) NULL,
  `camera_name` VARCHAR(255) NULL,
  `segment_id` VARCHAR(64) NULL,
  `segment_name` VARCHAR(255) NULL,
  `monitor_address` VARCHAR(255) NULL,
  `work_order_type` VARCHAR(64) NOT NULL,
  `work_order_describe` TEXT NOT NULL,
  `work_order_img_url` VARCHAR(1024) NOT NULL,
  `work_order_rank` INT NOT NULL DEFAULT 0,
  `work_order_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `work_order_stage` VARCHAR(64) NOT NULL DEFAULT 'unassigned',
  `work_order_status` INT NOT NULL DEFAULT 0 COMMENT '0=unresolved, 1=resolved, 2=ignored',
  `ai_suggestion` TEXT NULL,
  `scene_info` TEXT NULL,
  `completed_at` DATETIME NULL,
  `required_category` VARCHAR(64) NOT NULL DEFAULT 'traffic_police',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`work_order_id`),
  INDEX `idx_work_orders_rank` (`work_order_rank`),
  INDEX `idx_work_orders_status` (`work_order_status`),
  INDEX `idx_work_orders_stage` (`work_order_stage`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `mobile_reports` (
  `report_id` INT NOT NULL AUTO_INCREMENT,
  `reporter_user_id` INT NOT NULL,
  `title` VARCHAR(255) NOT NULL,
  `location` VARCHAR(255) NOT NULL,
  `detail` TEXT NOT NULL,
  `severity` VARCHAR(16) NOT NULL DEFAULT 'medium',
  `event_type` VARCHAR(64) NOT NULL DEFAULT 'other',
  `image_urls` TEXT NULL,
  `status` VARCHAR(16) NOT NULL DEFAULT 'pending',
  `work_order_id` INT NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`report_id`), INDEX `idx_mobile_reports_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `work_order_replies` (
  `work_order_reply_id` INT NOT NULL AUTO_INCREMENT,
  `work_order_id` INT NOT NULL,
  `work_order_reply_img_url` VARCHAR(1024) NOT NULL,
  `work_order_reply_msg` TEXT NOT NULL,
  `work_order_reply_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `work_order_reply_status` TINYINT(1) NOT NULL DEFAULT 0,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`work_order_reply_id`),
  INDEX `idx_work_order_replies_work_order_id` (`work_order_id`),
  CONSTRAINT `fk_work_order_replies_work_order`
    FOREIGN KEY (`work_order_id`) REFERENCES `work_orders` (`work_order_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `order_user` (
  `order_user_id` INT NOT NULL AUTO_INCREMENT,
  `work_order_id` INT NOT NULL,
  `user_id` INT NOT NULL,
  `order_user_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `order_user_status` VARCHAR(64) NOT NULL DEFAULT 'assigned',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`order_user_id`),
  UNIQUE KEY `uk_order_user_work_order_id` (`work_order_id`),
  INDEX `idx_order_user_user_id` (`user_id`),
  CONSTRAINT `fk_order_user_work_order`
    FOREIGN KEY (`work_order_id`) REFERENCES `work_orders` (`work_order_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_order_user_user`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
