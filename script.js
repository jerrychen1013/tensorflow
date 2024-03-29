/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 * 
 * 主要步驟
 * 1. 確認task
 *   (1) 為regression問題或是classification 問題？
 *   (2) 此學習為supervised 還是 unsupervised?
 *   (3) 輸入資料之shape是什麼？ 其輸出資料看起來像什麼？
 * 
 * 2. 準備資料
 *   (1) 可能時，清理資料並手動檢查
 *   (2) 訓練前務必shuffle資料
 *   (3) 常模化資料到神經網絡合理range，一般來說 0-1 or -1-1為最佳
 *   (4) 轉換資料到tensors
 * 
 * 3. 建立並跑model
 *   (1) 使用tf.sequential or tf.model定義model，並使用tf.layers.*加入layers
 *   (2) 選擇optimizer (adam是好工具)，以及參數（例如batch size & epochs數目）
 *   (3) 選擇適合的loss function來面對問題，以及精確的metric幫助評估進步。meanSquaredError常用在regression問題
 *   (4) 監督訓練，並看是否loss有下降
 * 
 * 4. 評估model
 *   選擇一個model的評估矩陣，使你可監測訓練。一但完成訓練，來做預測以看預測品質
 */




async function getData() {
    const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');  
    const carsData = await carsDataReq.json();  
    const cleaned = carsData.map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter(car => (car.mpg != null && car.horsepower != null));
    
    return cleaned;
  }

  async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    const values = data.map(d => ({
      x: d.horsepower,
      y: d.mpg,
    }));
  
    tfvis.render.scatterplot(
      {name: 'Horsepower v MPG'},
      {values}, 
      {
        xLabel: 'Horsepower',
        yLabel: 'MPG',
        height: 300
      }
    );
  
    // More code will be added below
    // Create the model
    const model = createModel();  
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;
        
    // Train the model  
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    // Make some predictions using the model and compare them to the
    // original data
    testModel(model, data, tensorData);
  }
  
  document.addEventListener('DOMContentLoaded', run);

  function createModel() {
    // Create a sequential model, 呼叫sequential model
    const model = tf.sequential(); 
    
    // Add a single hidden layer, dense階層用以矩陣加成input，再加數字(bias)到結果
    // inputShape是[1]，因為有1數字作為output
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
    
    // Add an output layer，units設定加權矩陣多大在階層，設為１代表每個input １加權
    model.add(tf.layers.dense({units: 1, useBias: true}));
  
    return model;
  }

  /**
 * Convert the input data to tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any 
  // intermediate tensors.
  
  return tf.tidy(() => {
    // Step 1. Shuffle the data, 幫助每批有不同資料變異性   
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    // 拿到２個array
    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg);

    // 轉換array到2d tensor, 其形狀為[num_examples, num_features_per_example]
    // 有input.length 範例，且每個範例有１個input特徵 (馬力)
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    // tensorflow 設計處理之數字不會太大，所以轉換為 0 - 1 / -1 - 1 最棒
    // 保持好習慣，在訓練資料前先常態化
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();  
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });  
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.  
  model.compile({
    optimizer: tf.train.adam(),  // 隨著example管理和update model
    loss: tf.losses.meanSquaredError, //使用mean square error比較model預測和真實value
    metrics: ['mse'],
  });
  
  const batchSize = 32;  // 每一次訓練遞迴中，model會看到多少資料subsets的大小
  const epochs = 50; // model去看全部資料的次數，這裡代表看dataset 50個遞迴
  
  return await model.fit(inputs, labels, {  // 開始訓練loop，當訓練結束後回傳promise
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(  // 產生繪圖loss & mse的function來監測訓練
      { name: 'Training Performance' },
      ['loss', 'mse'], 
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;  
  
  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling 
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    
    // 產生100個新example來喂model
    const xs = tf.linspace(0, 1, 100);      
    // 預測新example有相同形狀
    const preds = model.predict(xs.reshape([100, 1]));      
    
    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);
    
    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);
    
    // Un-normalize the data
    // dataSync(): 得到tensor內存的值(typedarray)
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });
  
 
  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });
  
  const originalPoints = inputData.map(d => ({
    x: d.horsepower, y: d.mpg,
  }));
  
  
  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'}, 
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
}