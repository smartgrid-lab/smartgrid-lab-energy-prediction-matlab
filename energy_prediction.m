
% energy_prediction.m
% تنبؤ استهلاك الطاقة باستخدام نموذج انحدار بسيط

% تحميل البيانات
data = readtable('data/energy_data.csv');

% تحويل التاريخ إلى رقم تسلسلي (اختياري)
data.Date = datenum(data.Date);

% تقسيم المتغيرات المستقلة والتابعة
X = [data.Temperature_C, data.Humidity_];
y = data.Energy_kWh;

% تقسيم البيانات إلى تدريب واختبار
cv = cvpartition(size(data,1), 'HoldOut', 0.3);
idxTrain = training(cv);
idxTest = test(cv);

XTrain = X(idxTrain,:);
yTrain = y(idxTrain);
XTest = X(idxTest,:);
yTest = y(idxTest);

% إنشاء نموذج انحدار خطي
mdl = fitlm(XTrain, yTrain);

% التنبؤ على بيانات الاختبار
yPred = predict(mdl, XTest);

% تقييم النموذج
rmse = sqrt(mean((yTest - yPred).^2));
r2 = mdl.Rsquared.Ordinary;

fprintf('RMSE: %.2f\n', rmse);
fprintf('R^2: %.2f\n', r2);

% رسم النتائج
figure;
plot(yTest, 'bo-'); hold on;
plot(yPred, 'r*-');
legend('Actual', 'Predicted');
xlabel('Test Sample');
ylabel('Energy Consumption (kWh)');
title('Actual vs Predicted Energy Consumption');
grid on;
