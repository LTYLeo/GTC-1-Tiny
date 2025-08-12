#include <bits/stdc++.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Windows.h>
#include <cmath>
#include <random>
#include <chrono>
#include <ctime>

#define ALL 1000001 // �ʻ���С
#define MAXN 10     // ����ά��
#define MAX_TEXT 100 // �����ı���󳤶�
#define MAX_INPUT 1024 // ������󳤶�
#define LEARNING_RATE 0.001 // ѧϰ��
#define LAMBDA 0.001 // ����ǿ��
#define MAX_GRADIENT 5.0 // �ݶȲü���ֵ
#define CONTEXT_LENGTH 1024 // �����ĳ����޸�Ϊ1024
#define COMMA_MARKER "<CMA>" // �����������

using namespace std;

// �� std::vector �����̬����
vector<vector<float>> WordVector(ALL, vector<float>(MAXN, 0.0f));
vector<vector<float>> Weights(ALL, vector<float>(MAXN, 0.0f));
vector<string> TokenList(ALL);
int alltoken = -1;

// ��������
void Dispose_wordvector(vector<string> line_data);
void Dispose_weights(vector<string> line_data);
void AddNewToken(const string& token);
float stringToFloat(const string& str);
float f1[MAXN], f2[MAXN];
float h1[MAXN], h2[MAXN];
double probability[ALL];
double pro1[ALL];
double pro2[ALL];
int cnt = -1;
int underprocess[CONTEXT_LENGTH]; // ������token����
int pos;
int currentIndex = 0; // ��ǰ��������

// �����������
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-0.1, 0.1); // �����ʼ����Χ

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// �滻�ַ����еĶ���Ϊ��Ƿ���
string replaceComma(const string& str) {
    string result = str;
    size_t pos = 0;
    while ((pos = result.find(",", pos)) != string::npos) {
        result.replace(pos, 1, COMMA_MARKER);
        pos += strlen(COMMA_MARKER);
    }
    return result;
}

// �滻�ַ����еı�Ƿ���Ϊ����
string restoreComma(const string& str) {
    string result = str;
    size_t pos = 0;
    while ((pos = result.find(COMMA_MARKER, pos)) != string::npos) {
        result.replace(pos, strlen(COMMA_MARKER), ",");
        pos++;
    }
    return result;
}

// �ַ���ת������
float stringToFloat(const string& str) {
    return stof(str);
}

// ��־��¼����
void LogProgress(const string& logPath, const string& message) {
    ofstream logFile(logPath, ios_base::app);
    if (!logFile.is_open()) {
        cerr << "�޷�д����־�ļ���" << endl;
        return;
    }

    logFile << message << endl;
    logFile.close();
}

// �������
void SaveCheckpoint(const string& checkpointPath, const string& wvpath, const string& wghspath) {

    ofstream checkpointFile(checkpointPath);
    if (!checkpointFile.is_open()) {
        cerr << "�޷���������ļ���" << endl;
        return;
    }

    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    checkpointFile << "Checkpoint at " << ctime(&now_time);

    checkpointFile.close();
    cout << "���㱣��ɹ���" << endl;
}

void Multi_Point_Attention() {
    // Feature 1
    memset(f1, 0, sizeof(f1));
    for (int i = 0; i < MAXN; i++) {
        for (int j = 0; j <= cnt; j++) {
            f1[i] += Weights[underprocess[j]][i] * WordVector[underprocess[j]][i];
        }
    }
    for (int i = 0; i < MAXN; i++) {
        f1[i] /= 100;
    }
    for (int i = 0; i < MAXN; i++) {
        h1[i] = sigmoid(f1[i]);
    }

    // Feature 2
    memset(f2, 0, sizeof(f2));
    for (int i = 0; i < MAXN; i++) {
        for (int j = 0; j <= cnt; j++) {
            for (int k = 0; k <= cnt; k++) {
                f2[i] += Weights[underprocess[j]][i] * WordVector[underprocess[k]][i];
            }
        }
    }
    for (int i = 0; i < MAXN; i++) {
        f2[i] /= (cnt + 1);
    }
    for (int i = 0; i < MAXN; i++) {
        f2[i] /= 100;
    }
    for (int i = 0; i < MAXN; i++) {
        h2[i] = sigmoid(f2[i]);
    }
}

void Softmax() {
    // pro1
    double sum_exp = 0;
    for (int i = 0; i <= alltoken; i++) {
        double dot_product = 0;
        for (int j = 0; j < MAXN; j++) {
            dot_product += h1[j] * WordVector[i][j];
        }
        pro1[i] = exp(dot_product);
        sum_exp += pro1[i];
    }
    for (int i = 0; i <= alltoken; i++) {
        pro1[i] /= sum_exp;
    }

    // pro2
    sum_exp = 0;
    for (int i = 0; i <= alltoken; i++) {
        double dot_product = 0;
        for (int j = 0; j < MAXN; j++) {
            dot_product += h2[j] * WordVector[i][j];
        }
        pro2[i] = exp(dot_product);
        sum_exp += pro2[i];
    }
    for (int i = 0; i <= alltoken; i++) {
        pro2[i] /= sum_exp;
    }

    // probability
    for (int i = 0; i <= alltoken; i++) {
        probability[i] = (pro1[i] + pro2[i]) / 2;
    }
}

void Dispose_wordvector(vector<string> line_data) {
    int tot = 0;
    int position;
    for (string str : line_data) {
        tot++;
        if (tot == 1) {
            position = stoi(str);
            continue;
        } else if (tot == 2) {
            TokenList[position] = restoreComma(str);
            continue;
        } else {
            WordVector[position][tot - 3] = stringToFloat(str); // ʹ���º���
            continue;
        }
    }
    alltoken++;
}

void Dispose_weights(vector<string> line_data) {
    int tot = 0;
    int position;
    for (string str : line_data) {
        tot++;
        if (tot == 1) {
            position = stoi(str);
            continue;
        } else {
            Weights[position][tot - 2] = stringToFloat(str); // ʹ���º���
            continue;
        }
    }
}

void AddNewToken(const string& token) {
    alltoken++;
    TokenList[alltoken] = token;

    // �����ʼ�� WordVector
    for (int j = 0; j < MAXN; j++) {
        WordVector[alltoken][j] = dis(gen);
    }

    // �����ʼ�� Weights
    for (int j = 0; j < MAXN; j++) {
        Weights[alltoken][j] = dis(gen);
    }
}

void SaveModel(const string& wvpath, const string& wghspath) {
    ofstream wvfile(wvpath);
    if (!wvfile.is_open()) {
        cout << "�޷�����������ļ���" << endl;
        return;
    }

    // ���� WordVector�����ָ�ʽһ��
    for (int i = 0; i <= alltoken; i++) {
        wvfile << i << "," << replaceComma(TokenList[i]); // д��index��token
        for (int j = 0; j < MAXN; j++) {
            wvfile << "," << WordVector[i][j];
        }
        wvfile << endl;
    }
    wvfile.close();
    cout << wvpath << " ����ɹ���" << endl;

    ofstream wghsfile(wghspath);
    if (!wghsfile.is_open()) {
        cout << "�޷�����Ȩ���ļ���" << endl;
        return;
    }

    // ���� Weights�����ָ�ʽһ��
    for (int i = 0; i <= alltoken; i++) {
        wghsfile << i; // д��index
        for (int j = 0; j < MAXN; j++) {
            wghsfile << "," << Weights[i][j];
        }
        wghsfile << endl;
    }
    wghsfile.close();
    cout << wghspath << " ����ɹ���" << endl;
}

int main() {
    SetConsoleTitle("GTC-1 Tiny (gtc-1-tiny-build-005-CMAfixed)");
    cout << "Loading Data..." << endl;
    string wvpath = "WordVector_Tiny.csv";
    string wghspath = "Weights_Tiny.csv";
    string logPath = "training_log.txt";
    string checkpointPath = "checkpoint.txt";

    ifstream wvfile(wvpath, ios::in);
    ifstream wghsfile(wghspath, ios::in);

    // ��ȡ������
    if (!wvfile.is_open()) {
        cout << "Data Crush!" << endl;
        exit(1);
    } else {
        string line;
        vector<string> words;
        string word;
        getline(wvfile, line);
        istringstream strin;
        while (getline(wvfile, line)) {
            words.clear();
            strin.clear();
            strin.str(line);
            while (getline(strin, word, ',')) {
                words.push_back(word);
            }
            Dispose_wordvector(words);
        }
        wvfile.close();
    }
    cout << wvpath << " Read OK." << endl;

    // ��ȡȨ��
    if (!wghsfile.is_open()) {
        cout << "Data Crush!" << endl;
        exit(1);
    } else {
        string line;
        vector<string> words;
        string word;
        getline(wghsfile, line);
        istringstream strin;
        while (getline(wghsfile, line)) {
            words.clear();
            strin.clear();
            strin.str(line);
            while (getline(strin, word, ',')) {
                words.push_back(word);
            }
            Dispose_weights(words);
        }
        wghsfile.close();
    }
    cout << wghspath << " Read OK." << endl;

    // ��ȡѵ������
    string trainFile = "TrainData.txt";
    ifstream trainData(trainFile);
    if (!trainData.is_open()) {
        cout << "�޷���ѵ���ļ���" << endl;
        exit(1);
    }

    string line;
    int tokenProcessedCount = 0; // �����������ڸ����Ѵ����token����
    while (getline(trainData, line)) {
        vector<string> tokens;
        istringstream ss(line);
        string token;
        
        // ʹ�ÿո�ָ�����ȡ�ǿ� token
        while (ss >> token) {
            if (!token.empty()) {
                tokens.push_back(token); // ֻ��ӷǿ� token
            }
        }

        cnt = tokens.size() - 1; // ��¼���� token ����
        for (int i = 0; i < tokens.size(); i++) {
            string s = tokens[i];
            
            // ��� token �Ƿ����� TokenList ��
            bool tokenExists = false;
            for (int j = 0; j <= alltoken; j++) {
                if (s == TokenList[j]) {
                    underprocess[currentIndex] = j; // ��������������
                    tokenExists = true;
                    break;
                }
            }
            
            // ��� token �����ڣ�������� token
            if (!tokenExists) {
                AddNewToken(s);
                underprocess[currentIndex] = alltoken; // ����ӵ� token ����
            }

            // Ԥ����һ�� token
            if (i > 0) { // ���ж���ѵ��
                double loss = 1.0;
                for (int step = 0; step < 6000; step++) {
                    Multi_Point_Attention();
                    Softmax();

                    // ������ʧ������ͨ����������ʧ
                    loss = -log(probability[underprocess[currentIndex]]);
                    // ���������ʧ
                    double regularization_loss = 0.0;
                    for (int j = 0; j < MAXN; j++) {
                        regularization_loss += (Weights[underprocess[currentIndex]][j] * Weights[underprocess[currentIndex]][j]);
                    }
                    loss += LAMBDA * regularization_loss;

                    cout << "Step: " << step + 1 << " Loss: " << loss << endl;

                    // ����ģ��Ȩ��
                    for (int j = 0; j < MAXN; j++) {
                        float gradient = probability[underprocess[currentIndex]] - 1; // �������Ǽ���������ݶ�
                        
                        // �ݶȲü�
                        if (gradient > MAX_GRADIENT) {
                            gradient = MAX_GRADIENT;
                        } else if (gradient < -MAX_GRADIENT) {
                            gradient = -MAX_GRADIENT;
                        }

                        Weights[underprocess[currentIndex]][j] -= LEARNING_RATE * (gradient + LAMBDA * Weights[underprocess[currentIndex]][j]);
                        WordVector[underprocess[currentIndex]][j] -= LEARNING_RATE * (gradient + LAMBDA * WordVector[underprocess[currentIndex]][j]);
                    }

                    // �����ʧС��0.1���˳�ѭ��
                    if (loss <= 0.1) {
                        break;
                    }
                }

                // �ҵ��������� token
                double maxx = -1;
                for (int i = 0; i <= alltoken; i++) {
                    if (probability[i] > maxx) {
                        maxx = probability[i];
                        pos = i;
                    }
                }
                // ������ɵ� token
                cout << " " << TokenList[pos];
                if (TokenList[pos] == "<TGC>") break; // ��������
                
                // ��������������
                currentIndex = (currentIndex + 1) % CONTEXT_LENGTH; // ѭ����������
                underprocess[currentIndex] = pos; // ���µ�ǰ�����ĵ����һ��token

                // ���´���� token ����
                tokenProcessedCount++;

                // ÿ5��token����һ�β����ͼ���
                if (tokenProcessedCount % 5 == 0) {
                	SaveModel(wvpath, wghspath);
                    SaveCheckpoint(checkpointPath, wvpath, wghspath);
                    string logMessage = "Processed " + to_string(tokenProcessedCount) + " tokens.";
                    LogProgress(logPath, logMessage);
                    tokenProcessedCount = 0;
                }
            }
        }
        cout << endl;
    }

    trainData.close();
    // ����ģ��
    SaveModel(wvpath, wghspath); // ������º��ģ�Ͳ���
    return 0;
}

