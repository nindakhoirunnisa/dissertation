import { 
    select, 
    json, 
    geoPath, 
    geoMercator, 
    zoom, 
    event,
    scaleSequential,
    max,
    sliderBottom,
    zoomIdentity,
    interpolateYlOrRd,
    interpolateCubehelixDefault,
    timeFormat,
    scaleLinear,
    scaleBand,
    scaleTime,
    axisBottom,
    axisLeft,
    scaleOrdinal,
    schemeTableau10 ,
    pie,
    arc,
    line,
    extent,
    timeMonth,
    bisector
} from 'd3';

let selectedProvince = [];

const svg = select("svg"),
				width = +svg.attr("width"),
				height = +svg.attr("height");

const gfg = geoMercator()
    .scale((width / 2.5 / Math.PI) * 10)
    .rotate([0, 0])
    .center([0, 0])
    .translate([-(width * 2.1), height / 3.8]);

const defs = svg.append('defs');
defs.append('pattern')
    .attr('id', 'stripes')
    .attr('patternUnits', 'userSpaceOnUse')
    .attr('width', 8)
    .attr('height', 8)
    .append('path')
    .attr('class', 'stripes')
    .attr('d', 'M-1,1 l2,-2 M0,8 l8,-8 M7,9 l2,-2')
    .attr('stroke-width', 0.5);

const g = svg.append('g');

let maxCount = 0; // Declare the variable to store the maximum count globally
const colorScale = scaleSequential(interpolateYlOrRd);

json('/get_max_count', function(data) {
    maxCount = data.max; // Update the global variable with the fetched maximum count
    colorScale.domain([0, maxCount]);
    createColorLegend();
    // ... rest of the code ...
    loadData(); // Call loadData here to ensure maxCount is updated before using it
});

const dateSlider = select('#dateSlider');
const sliderTooltip = select('#sliderTooltip');

let selectedDateString;

json('/get_date_range', function(data) {
    const startDate = new Date(data.min).getTime();
    const endDate = new Date(data.max).getTime();

    selectedDateString = data.max;

    const selectedDate = new Date(selectedDateString);
    const stringSelectedDate = selectedDate.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });

    let title = select('#bar-chart-title')

    title.selectAll('text').remove();
    title.append('text')
        .attr('class', 'map-title')
        .text(`Local News Headline's Sentiment About COVID-19 Vaccines on ${stringSelectedDate}`)

    loadData();

    dateSlider
        .attr('min', startDate)
        .attr('max', endDate)
        .attr('step', 24 * 60 * 60 * 100) // 1 day in milliseconds
        .property('value', endDate)
        .on('mousemove', handleSliderMouseMove)
        .on('input', () => {
            loadData();
        });

        function updateDateLabels() {
            // Update the min and max date labels
            select('.value-min').text(new Date(startDate).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric'
            })).style('font-size', '12px').style('font-weight', 'bold')
            select('.value-max').text(new Date(endDate).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric'
            })).style('font-size', '12px').style('font-weight', 'bold');
        }
        updateDateLabels();
});

function isMouseOverTooltip(mouseX, mouseY) {
    const tooltipRect = document.getElementById('dateSlider').getBoundingClientRect();
    return (
        mouseX >= tooltipRect.left &&
        mouseX <= tooltipRect.right &&
        mouseY >= tooltipRect.top &&
        mouseY <= tooltipRect.bottom
    );
}

function handleSliderMouseMove() {
    const mouseX = event.clientX;
    const mouseY = event.clientY;

    if (isMouseOverTooltip(mouseX, mouseY)) {
        const selectedTimestamp = +dateSlider.property('value');
        const selectedDate = new Date(selectedTimestamp);
        const selectedDateString = selectedDate.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
        
        // let title = select('#bar-chart-title')

        // title.selectAll('text').remove();
        // title.append('text')
        //     .attr('class', 'map-title')
        //     .text(`Local News Headline's Sentiment About COVID-19 Vaccines on ${selectedDateString}`)

        sliderTooltip
            .style('display', 'block')
            .text(`${selectedDateString}`)
            .style('left', `${mouseX-50}px`)
            .style('top', `${mouseY-525}px`)
    } else {
        sliderTooltip
            .style('display', 'none')
    }
    // Update the tooltip's content and position
}

let credibilityFilter = ['2']; // Default sentiment filter value

const dropdowns = document.querySelectorAll('.credibilityFilter')

dropdowns.forEach(dropdown => {
    const select = dropdown.querySelector('.select');
    const caret = dropdown.querySelector('.caret');
    const menu = dropdown.querySelector('.menu');
    const options = dropdown.querySelectorAll('.menu li')
    const selected = dropdown.querySelector('.selected')

    //Click event to element
    select.addEventListener('click', () => {
        //add the clicked styles to the select element
        select.classList.toggle('select-clicked');
        //add the rotate styles to the caret element
        caret.classList.toggle('caret-rotate')
        //add the open styles to the menu element
        menu.classList.toggle('menu-open')
    })

    options.forEach(option => {
        //add click event to the option element
        option.addEventListener('click', () => {
            //change selected inner text to clicked option inner text
            selected.innerText = option.innerText;
            //add the clicked select styles to the select element
            select.classList.remove('select-clicked');
            //add the rotate styles to caret
            caret.classList.remove('caret-rotate')
            //add open styles to menu
            menu.classList.remove('menu-open');
            //remove active class from all option
            options.forEach(option => {
                option.classList.remove('active')
            })
            //add active class to clicked option element
            option.classList.add('active')

            let value = option.getAttribute('value')
            credibilityFilter = [value];
            loadData()
        })
    })
})



json('get_geojson', function(data) {
    const checkboxesDiv = document.getElementById('provinceFilter');
    const provinceName = data.features.map(feature => feature.properties.name);
    const originalOrder = []

    provinceName.sort();

    provinceName.forEach((feature, index) => {
        const name = feature;
        const checkboxLabel = document.createElement('label')
        checkboxLabel.style.display = 'inline-flex';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.name = 'province';
        checkbox.value = name;
        checkbox.id = name;

        const label = document.createElement('span');
        label.textContent = name;

        originalOrder.push(index)

        checkbox.addEventListener('change', function() {
            if (this.checked) {
                selectedProvince.push(checkbox.value);
                checkboxLabel.style.fontWeight = 'bold';
                checkboxesDiv.removeChild(checkboxLabel);
                checkboxesDiv.insertBefore(checkboxLabel, checkboxesDiv.firstChild)
            } else {
                const selectedIndex = selectedProvince.indexOf(checkbox.value)
                if (selectedIndex !== -1) {
                    selectedProvince.splice(selectedIndex, 1);
                }
                checkboxLabel.style.fontWeight = 'normal';
                checkboxesDiv.removeChild(checkboxLabel);

                const adjustedIndex = originalOrder[index] < selectedProvince.length
                ? index + selectedProvince.length
                : index

                checkboxesDiv.insertBefore(checkboxLabel, checkboxesDiv.children[originalOrder[adjustedIndex]])
            }
            loadData();

        });

        checkboxLabel.appendChild(checkbox);
        checkboxLabel.appendChild(label);
        checkboxesDiv.appendChild(checkboxLabel)
    });

})

let capturedSentimentMap;

function createColorLegend() {
    const legendData = colorScale.ticks(8).reverse(); // Adjust the number of legend segments as needed    
    const colorLegendContainer = select('#color-legend-container');

    const colorLegend = colorLegendContainer.append('svg')
        .attr('class', 'color-legend')
        .attr('width', legendData.length * 70)
        .attr('height', 50)
        .attr('transform', 'translate(10, -55)'); // Adjust the positioning as needed

    const legendItems = colorLegend.selectAll('.legend-item')
        .data(legendData)
        .enter().append('g')
        .attr('class', 'legend-item')
        // Adjust the spacing between items

    legendItems.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 50)
        .attr('height', 15)
        .style('stroke', 'gray')
        .style('stroke-width', 0.5)
        .style('fill', (d, i)=> {
            return i === 8 ? 'url(#stripes)' : colorScale(d) 
        }) 
        .attr('transform', (d, i) => `translate(${i * 50},0)`);

    legendItems.append('text')
        .attr('x', 7)
        .attr('y', 30) // 32Adjust the text position as needed
        .style('font-size', '12px')
        .style('fill', '#333')
        .style('font-weight', 'bold')
        .text( (d,i) => {
            console.log(d)
            if ( i === 8) {
                return 'No data'
            }
            return `${d-1} - ${d}`
        })
        .attr('transform', (d, i) => {
            if (i === 1) {
                return `translate(${i * 49},0)`
            } else if (i === 2) {
                return `translate(${i * 50},0)`
            } else if (i ===3 ) {
                return `translate(${i * 51},0)`
            } else if (i === 4) {
                return `translate(${i * 52},0)`
            } else if (i === 5) {
                return `translate(${i * 51},0)`
            } else if (i === 6) {
                return `translate(${i * 51},0)`
            } else if (i === 7) {
                return `translate(${i * 51},0)`
            } else {
                return `translate(${i * 50 - 3},0)`
            }
        });
    
    colorLegendContainer
        // .style('position', 'absolute')
        .style('top', '-160px')
        .style('right', '-10px')
}

function loadData() {
    json('/get_geojson', function(geoData) {
        const credibilityQuery = credibilityFilter.join(',');
        const selectedTimestamp = +dateSlider.property('value');
        const selectedDate = new Date(selectedTimestamp);
        const mapTooltip = select('#map-tooltip')
        selectedDateString = selectedDate.toISOString().split('T')[0];
        const stringSelectedDate = selectedDate.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    
        let title = select('#bar-chart-title')
    
        title.selectAll('text').remove();
        title.append('text')
            .attr('class', 'map-title')
            .text(`Local News Headline's Sentiment About COVID-19 Vaccines on ${stringSelectedDate}`)
        
        json(`/get_news_count?credibility=${credibilityQuery}&date=${selectedDateString}`, function(sentimentResponse) {
            const sentimentData = Array.isArray(sentimentResponse) ? sentimentResponse : [];
            const sentimentMap = new Map(sentimentData.map(d => [d._id, d]));
            capturedSentimentMap = sentimentMap;
            const colorValue = d => sentimentMap.get(d.properties.name)?.count || 0;

            //Bind data and update existing paths
            const paths = g.selectAll('path').data(geoData.features);

            //Update existing paths
            paths  
                .attr('fill', d => {
                    const dataValue = colorValue(d);
                    return dataValue === 0 ? 'url(#stripes)': colorScale(dataValue);
                })
                .attr('opacity', d => 
                (!selectedProvince.length || selectedProvince.includes(d.properties.name))
                ? 1
                : 0.2
                )
                .transition().duration(500)
                .on('mouseover', function(d) {
                    let sentiment;
                    const mouseX = event.clientX;
                    const mouseY = event.clientY;
                    capturedSentimentMap.forEach((value, key) => {
                        if (key === d.properties.name) {
                            sentiment = value;
                        }
                    });
                    if (sentiment) {
                        const { count, negativeCount, neutralCount, positiveCount } = sentiment;
                        mapTooltip
                            .style('display', 'block')
                            .transition().duration(500)
                            .style('left', `${mouseX}px`)
                            .style('top', `${mouseY}px`)
                            .style('font-size', '40px')
                            .html(`<strong>Province:</strong> ${d.properties.name}<br><strong>Total Articles:</strong> ${count}<br><strong>Positive Articles:</strong> ${positiveCount}<br><strong>Neutral Articles:</strong> ${neutralCount}<br><strong>Negative Articles:</strong> ${negativeCount}`)
                    } else {
                        mapTooltip
                            .style('display', 'block')
                            .transition().duration(500)
                            .style('left', `${mouseX}px`)
                            .style('top', `${mouseY}px`)
                            .style('font-size', '20px')
                            .html(`<strong>Province:</strong> ${d.properties.name}<br>No Data`)
                    }
                })
                .on('mouseout', function() {
                    mapTooltip
                        .style('display', 'none')
                        .transition().duration(500)
                })
            
            //Enter new paths
            paths.enter()
                .append('path')
                    .attr('class', 'province')
                    .attr('d', geoPath().projection(gfg))
                    .attr('fill', d => {
                        const dataValue = colorValue(d);
                        return dataValue === 0 ? 'url(#stripes)': colorScale(dataValue);
                    })
                    .on('mouseover', function(d) {
                        let sentiment;
                        const mouseX = event.clientX;
                        const mouseY = event.clientY;
                        capturedSentimentMap.forEach((value, key) => {
                            if (key === d.properties.name) {
                                sentiment = value;
                            }
                        });
                        if (sentiment) {
                            const { count, negativeCount, neutralCount, positiveCount } = sentiment;
                            mapTooltip
                                .style('display', 'block')
                                .style('left', `${mouseX + 5}px`)
                                .style('top', `${mouseY + 5}px`)
                                .style('font-size', '10px')
                                .html(`<strong>Province:</strong> ${d.properties.name}<br><strong>Total Articles:</strong> ${count}<br><strong>Positive Articles:</strong> ${positiveCount}<br><strong>Neutral Articles:</strong> ${neutralCount}<br><strong>Negative Articles:</strong> ${negativeCount}`)
                        } else {
                            mapTooltip
                                .style('display', 'block')
                                .style('left', `${mouseX + 5}px`)
                                .style('top', `${mouseY + 5}px`)
                                .style('font-size', '10px')
                                .html(`<strong>Province:</strong> ${d.properties.name}<br>No Data`)
                        }
                    })
                    .on('mouseout', function() {
                        mapTooltip
                            .style('display', 'none')
                    })
                    // .append('title')
                    //     .text(d => {
                    //         const sentiment = sentimentMap.get(d.properties.name);
                    //         if (sentiment) {
                    //             const { count, negativeCount, neutralCount, positiveCount } = sentiment;
                    //             return `${d.properties.name}:\nTotal Articles: ${count}\nNegative: ${negativeCount}\nNeutral: ${neutralCount}\nPositive: ${positiveCount}`;
                    //         }
                    //         return `${d.properties.name}: \nNo data`;
                    //     })
                    //     .attr('class', 'tooltip')
        });
    });
}

//BAR CHART
const margin = { top: 50, right: 50, bottom: 20, left: 100 };
const widthN = 500 - margin.left - margin.right;
const heightN = 250 - margin.top - margin.bottom;

// Create a new SVG element for the bar chart
const barSvg = select("#bar-container")
    .append("svg")
    .attr("width", widthN + margin.left + margin.right)
    .attr("height", heightN + margin.top + margin.bottom)
    .attr("id", "bar-svg"); // Add an ID or class to select later if needed

// Append the bar chart SVG container to the new SVG element
const barChart = barSvg.append("g")
    .attr("id", "bar-container")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Fetch data for the bar chart and create the bar chart
json('/get_most_negative_provs', function(barChartData) {
    createBarChart(barChartData);
});

// Function to create the bar chart
function createBarChart(data) {
    // Create scales for x and y axes
    const x = scaleLinear().range([0, widthN]);
    const y = scaleBand().range([0, heightN]).padding(0.1);
    const barColorScale = scaleOrdinal(schemeTableau10 )

    const xAxis = axisBottom(x)
        .tickSize(0)

    const yAxis = axisLeft(y)
        .tickSize(0)
        .tickPadding(10);
    
    barChart.selectAll('line.vertical-grid')
        .data(x.ticks(5))
        .enter()
        .append('line')
        .attr('class', 'vertical-grid')
        .attr('x1', function (d) { return x(d);})
        .attr('y1', 0)
        .attr('x2', function (d) { return x(d);})
        .attr('y2', heightN)
        .style('stroke', 'gray')
        .style('stroke-width', 0.5)
        .style('stroke-dasharray', '3 3')

    // Update the domain of x and y scales based on the data
    x.domain([0, max(data, d => d.total)]);
    y.domain(data.map(d => d.province));

    // Select all the bars and their corresponding labels
    const bars = barChart.selectAll('.bar');
    const labels = barChart.selectAll('.label');
    
    bars
      .data(data)
      .enter().append('rect')
      .attr('class', 'bar')
      .attr('x', 0)
      .attr('y', d => y(d.province))
      .attr('width', d => x(d.total))
      .attr('height', y.bandwidth())
      .style('fill', (d, i) => barColorScale(i))
      
    labels
      .data(data)
      .enter().append('text')
      .attr('class', 'label')
      .attr('x', function (d) { return x(d.total) + 5; })
      .attr('y', function (d) { return y(d.province) + y.bandwidth() / 2; })
      .attr('dy', '.35em')
      .style('font-family', 'sans-serif')
      .style('font-size', '10px')
      .style('font-weight', 'bold')
      .style('fill', '#3c3d28')
      .text(function (d) { return d.total; });

    // Append x and y axes to the bar chart
    barChart.append('g')
        .attr('class', 'x axis')
        .style('font-size', '10px')
        .attr('transform', 'translate(0,' + heightN + ')')
        .call(xAxis)
        .call(g => g.select('.domain').remove());

    barChart.append('g')
        .attr('class', 'y axis')
        .style('font-size', '8px')
        .call(yAxis)
        .selectAll('path')
        .style('stroke-width', '1.75px')
        
    barChart.append('text')
        .attr('class', 'chart-title')
        .attr('x', 0 - (margin.left/2))
        .attr('y', 0 - (margin.top/2))
        .style('font-size', '12px')
        .style('font-weight', 'bold')
        .text('Top 5 Provinces with Negative Articles')
    
    // barChart.selectAll('.y.axis .tick text')
    // .text(function (d) {
    //     return d.toUpperCase();
    // })
}

//PIE CHART
json('/get_total_sentiment', function(data) {
    createPieChart(data)
})

function createPieChart(data) {
    const pieSvg = select('#pie-container')
        .append('svg')
        .attr('width', 350)
        .attr("height", heightN + margin.top + margin.bottom)
        .attr('id', 'pie-svg')
    const pieChart = pieSvg.append('g')
        .attr("id", "pie-container")
        .attr("transform", `translate(${widthN / 2}, ${(heightN + margin.top + margin.bottom) / 1.8})`); //or width/2, height/2
    const radius = Math.min(widthN, heightN) / 1.6
    const pieColorScale = scaleOrdinal(schemeTableau10 )
    const pieC = pie().value(d => d.count).sort(null)
    const arcC = arc().outerRadius(radius * 0.9).innerRadius(0)
    const hoverArc = arc().innerRadius(0).outerRadius(radius)
    const total = data.reduce((sum, d) => sum + d.count, 0);

    const g = pieChart.selectAll('.arc')
        .data(pieC(data))
        .enter().append('g')
        .attr('class', 'arc')
    
    g.append('path')
        .attr('d', arcC)
        .attr('class', 'arc')
        .style('fill', (d, i) => pieColorScale(i))
        .style('stroke', 'white')
        .style('fill-opacity', 0.8)
        .style('stroke-width', 1)
        .on('mouseover', function (d, i) {
            select(this)
                .style('fill-opacity', 1)
                .transition().duration(500)
                .attr('d', hoverArc)
        })
        .on('mouseout', function (d,i) {
            select(this)
                .style('fill-opacity', 0.8)
                .transition().duration(500)
                .attr('d', arcC)
        })
    
    g.append('text')
        .attr('transform', d=> `translate(${arcC.centroid(d)})`)
        .attr('dy', '0.35em')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('fill', 'white')
        .style('font-weight', 'bold')
        .text(d => {
            const percentage = (d.data.count /total) * 100
            return `${percentage.toFixed(2)}%`
        });

    //create color legend
    const legend = pieSvg.selectAll('.legend')
        .data(pieC(data))
        .enter()
        .append('g')
        .attr('class', 'legend')
        .attr('transform', (d, i) => `translate(-120, ${i * 20})`)

    legend.append('rect')
        .attr('x', widthN - 220) //ori: widthN - 18
        .attr('y', heightN + 5)
        .attr('width', 10)
        .attr('height', 10)
        .style('fill', (d,i) => pieColorScale(i))
    
    legend.append("text")
        .attr("x", widthN - 208)
        .attr("y", heightN + 10)
        .attr("dy", ".35em")
        .style("text-anchor", "start")
        .style("font-size", "10px")
        .style('fill', '#333')
        .text((d) => {
            if (d.data.sentiment === 1) {
                return 'Positive';
            } else if (d.data.sentiment === 0) {
                return 'Neutral';
            } else if (d.data.sentiment === -1) {
                return 'Negative';
            }
        });

    pieChart.append('text')
        .attr('class', 'chart-title')
        .attr('x', -150)
        .attr('y', -115)
        .style('font-size', '12px')
        .style('font-weight', 'bold')
        .text('Sentiment of Published Articles')    
}

function formatDateToMonthYear(date) {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const year = date.getFullYear();
    const month = months[date.getMonth()];
    return `${month} ${year}`;
  }

//LINE CHART
json('/get_sentiment_count_each_date', function(data) {
    const x = scaleTime()
        .range([0, widthN])
    const y = scaleLinear()
        .range([heightN, 0])

        
    const lines = line()
        .x(d => x(d.year_month))
        .y(d => y(d.count))

    const svgLine = select('#line-container')
        .append('svg')
            .attr('width', 365)
            .attr('height', heightN + margin.top + margin.bottom)
        .append('g')
            .attr('transform', `translate(${margin.left/2.2},${margin.top})`);
    
    data.forEach(function(d) {
        const yearMonthArray = d.year_month.split('-');
        const year = parseInt(yearMonthArray[0]);
        const month = parseInt(yearMonthArray[1]); // JavaScript months are 0-indexed

        d.year_month  = new Date(year, month);
        d.year = new Date(year)
    });

    const lineTooltip = select('#line-tooltip')
        .attr('class', 'line-tooltip')
    
    
    x.domain(extent(data, d => d.year_month));
    y.domain([0, max(data, d => d.count)]);

    //x axis
    svgLine.append('g')
        .attr('transform', `translate(0,${heightN})`)
        .call(axisBottom(x)
            .ticks(timeMonth.every(6)))
        .call(g => g.select('.domain').remove())
        .selectAll('.tick line')
        .style('stroke-opacity', 0)
        .style('fill', '#777')
    svgLine.selectAll('.tick text')
        .attr('fill', '#777')

    //y axis
    svgLine.append('g')
        .call(axisLeft(y))
        .call(g => g.select('.domain').remove())
        .selectAll('.tick line')
        .style('stroke-opacity', 0)
    svgLine.selectAll('.tick text')
        .attr('fill', '#777') 
    
    const path = svgLine.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', 'steelblue')
        .attr('stroke-width', 1.5)
        .attr('d', lines)

    svgLine.append('text')
        .attr('class', 'shart-title')
        .attr('x', -15)
        .attr('y', -25)
        .style('font-size', '12px')
        .style('font-weight', 'bold')
        .text('Number of Published Articles Over Time')

    const circle = svgLine.append('circle')
        .attr('r', 0)
        .attr('fill', 'steelblue')
        .style('stroke', 'white')
        .attr('opacity', .70)
        .style('pointer-events', 'none')
    
    const listeningRect = svgLine.append('rect')
        .attr('width', widthN)
        .attr('height', heightN)
        .attr('class', 'rectline')
        
    listeningRect.on('mousemove', function () {
        const xCoord = event.clientX - this.getBoundingClientRect().left;
        const bisectDate = bisector(d => d.year_month).left;
        const x0 = x.invert(xCoord);
        const i = bisectDate(data, x0, 1);
        const d0 = data[i - 1];
        const d1 = data[i];
        const d = x0 - d0.year_month > d1.year_month - x0 ? d1 : d0;
        const xPos = x(d.year_month);
        const yPos = y(d.count);

        circle.attr('cx', xPos)
            .attr('cy', yPos)

        circle.transition()
            .duration(50)
            .attr('r', 5)
    
        lineTooltip
            .style('display', 'block')
            .style('left', `${xPos+885}px`)
            .style('top', `${yPos+610}px`)
            .style('font-size', '10px')
            .html(`<strong>Date:</strong> ${formatDateToMonthYear(d.year_month)}<br><strong>Articles:</strong> ${d.count}`)
    });

    listeningRect.on('mouseleave', function() {
        circle.transition()
            .duration(50)
            .attr('r',0)
        lineTooltip.style('display', 'none')
    })
})
