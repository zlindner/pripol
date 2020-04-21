import React, { useState } from 'react';
import GlobalStyle from './globalStyle';
import Landing from './components/landing';
import Analysis from './components/analysis';

interface IPrediction {
    segment: string;
    dataPractice: number;
}

const App = () => {
    const [predictions, setPredictions] = useState<IPrediction[] | []>([]);
    const [analyzed, setAnalyzed] = useState(false);

    const showAnalysis = (data: { segment: string; data_practice: string }[]) => {
        let predictions: IPrediction[] = [];

        // convert string dataPractice to number
        data.forEach((d) => predictions.push({ segment: d.segment, dataPractice: +d.data_practice }));

        setPredictions(predictions);
        setAnalyzed(true);
    };

    return (
        <React.Fragment>
            <GlobalStyle />
            <Landing showAnalysis={showAnalysis} />
            {analyzed && <Analysis predictions={predictions} />}
        </React.Fragment>
    );
};

export default App;
