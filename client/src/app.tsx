import React, { useState } from 'react';
import GlobalStyle from './globalStyle';
import Landing from './components/landing';
import Analysis from './components/analysis';

interface ISegment {
    segment: string;
    data_practice: string;
}

const App = () => {
    const [segments, setSegments] = useState<ISegment[] | []>([]);
    const [analyzed, setAnalyzed] = useState(false);

    const showAnalysis = (segments: ISegment[]) => {
        setSegments(segments);
        setAnalyzed(true);
    };

    return (
        <React.Fragment>
            <GlobalStyle />
            <Landing showAnalysis={showAnalysis} />
            {analyzed && <Analysis segments={segments} />}
        </React.Fragment>
    );
};

export default App;
