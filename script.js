let csvData = [];
let r2Scores = { RNN: null, LSTM: null, FNN: null };

const output = document.getElementById('output');
const result = document.getElementById('result');
const dataTable = document.getElementById('dataTable');

function logout() {
  alert("Logged out successfully.");
  window.location.href = "login.html";
}

function handleUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  document.getElementById("fileNameDisplay").textContent = "Selected File: " + file.name;

  Papa.parse(file, {
    header: true,
    dynamicTyping: true,
    complete: function(results) {
      csvData = results.data.filter(row => Object.values(row).some(cell => cell !== ""));
      displayTable(csvData);
      output.innerText = "📄 File loaded successfully.\n";
    }
  });
}

function displayTable(data) {
  if (!data.length) {
    dataTable.innerHTML = "";
    return;
  }
  
  const headers = Object.keys(data[0]);
  const thead = `<tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>`;
  const tbody = data.map(row => 
    `<tr>${headers.map(h => `<td>${row[h]}</td>`).join('')}</tr>`
  ).join('');
  
  dataTable.innerHTML = thead + tbody;
}

function processData() {
  if (!csvData.length) {
    alert("Please upload a dataset first.");
    return;
  }
  output.innerText += "🛠 Data preprocessing simulated...\n";
}

function encodeText(data, key) {
  const unique = [...new Set(data.map(row => row[key]))];
  const map = Object.fromEntries(unique.map((val, i) => [val, i]));
  return data.map(row => map[row[key]]);
}

async function trainModel(type) {
  if (!csvData.length) {
    alert("Please upload a dataset first.");
    return;
  }

  const filteredData = csvData.filter(d => d.place && d.rainfall != null && d.temperature != null && d.weather && d.yield != null);
  if (!filteredData.length) {
    alert("Dataset missing required columns (place, rainfall, temperature, weather, yield)!");
    return;
  }

  const placeEncoded = encodeText(filteredData, 'place');
  const weatherEncoded = encodeText(filteredData, 'weather');

  const inputs = filteredData.map((d, i) => [placeEncoded[i], d.rainfall, d.temperature, weatherEncoded[i]]);
  const labels = filteredData.map(d => [d.yield]);

  const inputTensor = tf.tensor2d(inputs);
  const labelTensor = tf.tensor2d(labels);

  const inputMax = inputTensor.max(0), inputMin = inputTensor.min(0);
  const labelMax = labelTensor.max(0), labelMin = labelTensor.min(0);

  const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
  const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

  const reshapedInputs = normalizedInputs.reshape([inputs.length, 1, 4]);

  const model = tf.sequential();
  model.add(type === "LSTM" 
    ? tf.layers.lstm({ units: 32, inputShape: [1, 4], activation: 'tanh' })
    : tf.layers.simpleRNN({ units: 16, inputShape: [1, 4], activation: 'relu' })
  );
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  output.innerText += `🧠 Training ${type} model...\n`;

  await model.fit(reshapedInputs, normalizedLabels, {
    epochs: 50,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        output.innerText += `Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}\n`;
      }
    }
  });

  const predictions = model.predict(reshapedInputs);
  const unNormalizedPreds = predictions.mul(labelMax.sub(labelMin)).add(labelMin);
  const predsArray = await unNormalizedPreds.array();
  const labelsArray = labels;

  const r2 = calculateR2(predsArray, labelsArray);
  r2Scores[type] = r2.toFixed(3);

  output.innerText += `\n🌟 Predicted yield (${type}) for first row: ${predsArray[0][0].toFixed(2)} tons/ha\n`;
  output.innerText += `📊 R² Score (${type}): ${r2Scores[type]}\n`;
}

async function predictWithFNN() {
  if (!csvData.length) {
    alert("Please upload a dataset first.");
    return;
  }

  const filteredData = csvData.filter(d => d.rainfall != null && d.temperature != null && d.SoilQuality != null && d.yield != null);
  if (!filteredData.length) {
    alert("Dataset must contain rainfall, temperature, SoilQuality, and yield!");
    return;
  }

  const inputs = filteredData.map(d => [d.rainfall, d.temperature, d.SoilQuality]);
  const labels = filteredData.map(d => [d.yield]);

  const inputTensor = tf.tensor2d(inputs);
  const labelTensor = tf.tensor2d(labels);

  const inputMax = inputTensor.max(0), inputMin = inputTensor.min(0);
  const labelMax = labelTensor.max(0), labelMin = labelTensor.min(0);

  const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
  const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [3] }));
  model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  output.innerText += `🧠 Training Feedforward Neural Network (FNN)...\n`;

  await model.fit(normalizedInputs, normalizedLabels, {
    epochs: 50,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        output.innerText += `Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}\n`;
      }
    }
  });

  const predictions = model.predict(normalizedInputs);
  const unNormalizedPreds = predictions.mul(labelMax.sub(labelMin)).add(labelMin);
  const predsArray = await unNormalizedPreds.array();
  const labelsArray = labels;

  const r2 = calculateR2(predsArray, labelsArray);
  r2Scores["FNN"] = r2.toFixed(3);

  displayPredictionTable(filteredData, predsArray);

  output.innerText += `🌟 FNN prediction completed.\n📊 R² Score (FNN): ${r2Scores.FNN}\n`;
}

function calculateR2(predsArray, labelsArray) {
  const ssRes = predsArray.reduce((sum, p, i) => sum + Math.pow(p[0] - labelsArray[i][0], 2), 0);
  const meanLabel = labelsArray.reduce((sum, l) => sum + l[0], 0) / labelsArray.length;
  const ssTot = labelsArray.reduce((sum, l) => sum + Math.pow(l[0] - meanLabel, 2), 0);
  return 1 - (ssRes / ssTot);
}

function displayPredictionTable(data, predsArray) {
  let html = `<table><thead><tr>
    <th>Rainfall</th><th>Temperature</th><th>Soil Quality</th><th>Predicted Crop Yield</th>
  </tr></thead><tbody>`;

  data.forEach((row, i) => {
    html += `<tr>
      <td>${row.rainfall}</td>
      <td>${row.temperature}</td>
      <td>${row.SoilQuality}</td>
      <td class="prediction">${predsArray[i][0].toFixed(2)}</td>
    </tr>`;
  });

  html += '</tbody></table>';
  result.innerHTML = html;
}

function displayChart() {
  const ctx = document.getElementById('accuracyChart').getContext('2d');

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['RNN', 'LSTM', 'FNN'],
      datasets: [{
        label: 'R² Score',
        data: [
          parseFloat(r2Scores.RNN) || 0,
          parseFloat(r2Scores.LSTM) || 0,
          parseFloat(r2Scores.FNN) || 0
        ],
        backgroundColor: ['#4caf50', '#2980b9', '#f39c12']
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 1
        }
      }
    }
  });
}
