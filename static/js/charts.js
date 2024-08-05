function createChart(chartId, chartType, chartData, options) {
    const ctx = document.getElementById(chartId).getContext('2d');
    new Chart(ctx, {
        type: chartType,
        data: chartData,
        options: options
    });
}

function addChartExplanation(chartId, explanation) {
    const chartContainer = document.getElementById(chartId).parentElement;
    const explanationElement = document.createElement('p');
    explanationElement.className = 'mt-3 text-muted';
    explanationElement.textContent = explanation;
    chartContainer.appendChild(explanationElement);
}