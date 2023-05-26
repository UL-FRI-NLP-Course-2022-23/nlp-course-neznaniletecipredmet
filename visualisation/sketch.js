
let characters = [];
let sentiment = [];
let protagonist = [];
let relationships = [];
let characterCoordinates = {};
let characterTraits = {};
let person;
let bubble;
let imgX;
let imgY;
let imgSize;
let current;

let primeColor = '#a90000';
let secondColor = '#898989';
let primeFont = 'Helvetica';
let secondFont = 'Arial';
let bubbleSize = 100;

function preload() {
    title = 'Zlata_ptica'
    let charactersTable = loadTable('./data/characters/' + title + '.csv', 'csv', () => {
        for(let i = 0; i < charactersTable.getRowCount(); i++) {
            characters.push(charactersTable.getString(i,0));
            sentiment.push(charactersTable.getString(i,1));
            protagonist.push(charactersTable.getString(i,2));

            let traits = []
            let j = 3
            while (charactersTable.get(i,j)) {
                if (!traits.includes(charactersTable.get(i,j))) {
                    traits.push(charactersTable.get(i,j));
                }
                j++;
            }
            characterTraits[characters[i]] = traits;
        }
    });

    let relationshipsTable = loadTable('data/relationships/' + title + '.csv', 'csv', () => {
        for(let i = 0; i < relationshipsTable.getRowCount(); i++) {
            let person1 = relationshipsTable.getString(i,0);
            let person2 = relationshipsTable.getString(i,1);
            let relation = relationshipsTable.getString(i,2);
            let percent = relationshipsTable.getString(i,3);
            if (percent > 0) {
                relationships.push([person1, person2, relation, percent]);
            }
        }
    });

    person = loadImage('img/person.png');
    bubble = loadImage('img/bubble.png');
}

function setup() {
    createVisualisation();
}

function draw() {
    fill('white');
    rect(0, 80, windowWidth, windowHeight-80);

    for (let i = 0; i < relationships.length; i++) {
        let char1 = relationships[i][0];
        let char2 = relationships[i][1];
        let type = relationships[i][2];
        let pow = relationships[i][3];
        let rel = 'nevtralno';
        let type_normalised = Math.abs(type) / 5
        let lineCenterX = (characterCoordinates[char1][0] + characterCoordinates[char2][0]) / 2;
        let lineCenterY = (characterCoordinates[char1][1] + characterCoordinates[char2][1]) / 2;
        let lineLength = Math.sqrt(Math.pow((characterCoordinates[char1][0] - characterCoordinates[char2][0]), 2) +
                         Math.pow((characterCoordinates[char1][1] - characterCoordinates[char2][1]), 2));
        let lineAngle = Math.asin(Math.abs(characterCoordinates[char1][1] - characterCoordinates[char2][1])/lineLength);
        let minX = char1;
        let minY = char1;
        if (characterCoordinates[char1][0] > characterCoordinates[char2][0]) minX = char2;
        if (characterCoordinates[char1][1] > characterCoordinates[char2][1]) minY = char2;

        stroke(200);
        if (type > 1) {rel = "zaveznik";stroke(100, 255, 100, 100 - 100*Math.abs(type_normalised));}
        if (type < -1) {rel = "sovražnik";stroke(255, 100, 100, 100 - 100*Math.abs(type_normalised));}

        strokeWeight(3 + 7*pow);
        line(characterCoordinates[char1][0], characterCoordinates[char1][1], characterCoordinates[char2][0], characterCoordinates[char2][1]);

        fill('black');
        noStroke();
        textSize(16);
        //let rotationAxis = createVector(characterCoordinates[max][0]-characterCoordinates[min][0], characterCoordinates[max][1] - characterCoordinates[min][1])
        //rotate(lineAngle);
        textAlign(CENTER, CENTER);
        //translate(lineCenterX,lineCenterY);
        text(rel, characterCoordinates[minX][0], characterCoordinates[minY][1], Math.abs(characterCoordinates[char1][0] - characterCoordinates[char2][0]), Math.abs(characterCoordinates[char1][1] - characterCoordinates[char2][1]))
        //text(rel, 300, 300, 500, 500);
        //text(rel, lineCenterX, lineCenterY, 300, 30);
        //text(rel, characterCoordinates[char1][0], characterCoordinates[char1][1]-8, lineLength, 16);
        //translate(-lineCenterX,-lineCenterY);
        //fill('red');
        //rect(characterCoordinates[min][0], characterCoordinates[min][1], Math.abs(characterCoordinates[char1][0] - characterCoordinates[char2][0]), Math.abs(characterCoordinates[char1][1] - characterCoordinates[char2][1]));
        //rotate(-lineAngle);
    }
    fill('black');
    noStroke();
    textSize(16);
    textAlign(CENTER, CENTER);
    for (let i = 0; i < characters.length; i++) {
        imgX = characterCoordinates[characters[i]][0];
        imgY = characterCoordinates[characters[i]][1];
        imgSize = characterCoordinates[characters[i]][2]*2;
        current = characters[i];

        image(person, imgX - imgSize/2, imgY - imgSize/2, imgSize, imgSize);
        text(characters[i].toUpperCase(), imgX - imgSize/2, imgY + imgSize/2, imgSize, 20);

        // if (mouseX > imgX - imgSize/2 && mouseX < imgX + imgSize/2 && mouseY > imgY - imgSize/2 && mouseY < imgY + imgSize/2) {
        if (characterTraits[characters[i]].length > 0) {
            tint(255, 100);
            image(bubble, imgX, imgY - imgSize/2 - bubbleSize, bubbleSize, bubbleSize);
            text(characterTraits[characters[i]].join("\n"), imgX, imgY - imgSize/2 - bubbleSize, bubbleSize, bubbleSize-10);
            tint(255, 255);
        }

        let sentiment_min = Math.min(...sentiment);
        let normalized_sentiment = (sentiment[i]-sentiment_min)/(Math.max(...sentiment)-sentiment_min);
        let pos_x = imgX - imgSize/2;
        if (normalized_sentiment > 0.75 || normalized_sentiment < 0.2) {pos_x = imgX + 7;}

        fill('red');
        if (sentiment[i] > 0.5) {fill('seagreen');}
        textSize(11);
        text(Math.round(sentiment[i]*100) + "%", pos_x, imgY - imgSize/2, imgSize/2, 20);

        textSize(16);
        fill('black');
    }
}

function windowResized() {
    createVisualisation();
}

function mousePressed() {
}

function mouseReleased() {
}

function mouseClicked() {
}

function createVisualisation() {
    createCanvas(windowWidth, windowHeight);
    background('white');

    textFont(primeFont);
    textSize(30);
    textAlign(CENTER);
    fill(primeColor);
    noStroke();
    rect(0, 25, width, 50);
    fill('white');
    text('Knowledge base from slovene fairytales', 0, 35, width, 75);

    let graphCenterX = windowWidth/2;
    let graphCenterY = windowHeight/2 - 50;
    let graphWidth = (windowWidth - 60)/2;
    let graphHeight = (windowHeight - 110)/2;
    let sentiment_min = Math.min(...sentiment);
    let normalize = (Math.max(...sentiment)-sentiment_min);

    for (let i = 0; i < characters.length; i++) {
        let imgSize = protagonist[i]*50 + 20;
        let r = Math.min(graphWidth, graphHeight) * (1 - protagonist[i]);
        let fi = Math.PI * ((sentiment[i]-sentiment_min)/normalize);
        let imgX = graphCenterX + r * Math.cos(fi);
        let imgY = graphCenterY + r * Math.sin(fi);
        characterCoordinates[characters[i]] = [imgX, imgY, imgSize];
    }
}
