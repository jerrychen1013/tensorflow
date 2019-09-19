/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
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