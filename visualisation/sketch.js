
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
    let charactersTable = loadTable('data/farytales_characters.csv', 'csv', 'header', () => {
        for(let i = 0; i < charactersTable.getRowCount(); i++) {
            characters.push(charactersTable.getString(i,0));
            sentiment.push(charactersTable.getString(i,1));
            protagonist.push(charactersTable.getString(i,2));

            let traits = []
            let j = 3
            while (charactersTable.get(i,j)) {
                traits.push(charactersTable.get(i,j));
                j++;
            }
            characterTraits[characters[i]] = traits;
        }
    });

    let relationshipsTable = loadTable('data/farytales_relationships.csv', 'csv', () => {
        for(let i = 0; i < relationshipsTable.getRowCount(); i++) {
            let person1 = relationshipsTable.getString(i,0);
            let person2 = relationshipsTable.getString(i,1);
            let relation = relationshipsTable.getString(i,2);
            let percent = relationshipsTable.getString(i,3);
            relationships.push([person1, person2, relation, percent]);
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
        let rel = relationships[i][2];
        let pow = relationships[i][3];
        let lineCenterX = (characterCoordinates[char1][0] + characterCoordinates[char2][0]) / 2;
        let lineCenterY = (characterCoordinates[char1][1] + characterCoordinates[char2][1]) / 2;
        let lineLength = Math.sqrt(Math.pow((characterCoordinates[char1][0] - characterCoordinates[char2][0]), 2) +
                         Math.pow((characterCoordinates[char1][1] - characterCoordinates[char2][1]), 2));
        let lineAngle = Math.asin(Math.abs(characterCoordinates[char1][1] - characterCoordinates[char2][1])/lineLength);
        let minX = char1;
        let minY = char1;
        if (characterCoordinates[char1][0] > characterCoordinates[char2][0]) minX = char2;
        if (characterCoordinates[char1][1] > characterCoordinates[char2][1]) minY = char2;

        stroke(100 + 100*pow);
        strokeWeight(10*pow);
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
    for (let i = 0; i < characters.length; i++) {
        imgX = characterCoordinates[characters[i]][0];
        imgY = characterCoordinates[characters[i]][1];
        imgSize = characterCoordinates[characters[i]][2]*2;
        current = characters[i];

        image(person, imgX - imgSize/2, imgY - imgSize/2, imgSize, imgSize);
        text(characters[i].toUpperCase(), imgX - imgSize/2, imgY + imgSize/2, imgSize, 20);

        if (mouseX > imgX - imgSize/2 && mouseX < imgX + imgSize/2 && mouseY > imgY - imgSize/2 && mouseY < imgY + imgSize/2) {
            tint(255, 100);
            image(bubble, imgX, imgY - imgSize/2 - bubbleSize, bubbleSize, bubbleSize);
            text(characterTraits[characters[i]].join("\n"), imgX, imgY - imgSize/2 - bubbleSize, bubbleSize, bubbleSize-10);
            tint(255, 255);
        }
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

    for (let i = 0; i < characters.length; i++) {
        let imgSize = protagonist[i]*50 + 20;
        let r = Math.min(graphWidth, graphHeight) * (1 - protagonist[i]);
        let fi = Math.PI * (1 - sentiment[i]);
        let imgX = graphCenterX + r * Math.cos(fi);
        let imgY = graphCenterY + r * Math.sin(fi);
        characterCoordinates[characters[i]] = [imgX, imgY, imgSize];
    }
}