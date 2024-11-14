#include <sqlite3.h>
#include <iostream>
#include <string>
#include <json/json.h>  // 用于JSON的操作

// 数据库操作类
class DatabaseManager {
public:
    DatabaseManager(const std::string& dbName) {
        // 打开或创建数据库
        if (sqlite3_open(dbName.c_str(), &db) != SQLITE_OK) {
            std::cerr << "Cannot open database: " << sqlite3_errmsg(db) << std::endl;
        }
        // else {
        //     std::cout << "Opened database successfully." << std::endl;
        // }
        createTable();
    }

    ~DatabaseManager() {
        // 关闭数据库
        sqlite3_close(db);
        // std::cout << "closed database successfully." << std::endl;
    }

    // 创建表
    void createTable() {
        const char* createTableSQL = R"(
            CREATE TABLE IF NOT EXISTS rtsp_logs (
                timestamp TEXT,
                ip_address TEXT,
                rtsp_url TEXT,
                data TEXT,
                PRIMARY KEY (ip_address, timestamp)
            );
        )";

        char* errorMessage = nullptr;
        if (sqlite3_exec(db, createTableSQL, nullptr, nullptr, &errorMessage) != SQLITE_OK) {
            std::cerr << "SQL Error: " << errorMessage << std::endl;
            sqlite3_free(errorMessage);
        }
        // else {
        //     std::cout << "Table created successfully or already exists." << std::endl;
        // }
    }

    // 插入数据
    void insertLog(const std::string& timestamp, const std::string& ip_address, const std::string& rtsp_url, const Json::Value& data) {
        std::string dataStr = data.toStyledString(); // 将 JSON 数据转换为字符串

        // 构造插入 SQL
        std::string insertSQL = "INSERT INTO rtsp_logs (timestamp, ip_address, rtsp_url, data) VALUES ('"
            + timestamp + "', '" + ip_address + "', '" + rtsp_url + "', '" + dataStr + "');";

        char* errorMessage = nullptr;
        if (sqlite3_exec(db, insertSQL.c_str(), nullptr, nullptr, &errorMessage) != SQLITE_OK) {
            std::cerr << "SQL Error: " << errorMessage << std::endl;
            sqlite3_free(errorMessage);
        }
        // else {
        //     std::cout << "Log inserted successfully." << std::endl;
        // }
    }

private:
    sqlite3* db;
};
