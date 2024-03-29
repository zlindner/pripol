import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import DataPractice from './dataPractice';
import Viewer from './viewer';

const Container = styled.div`
    width: 100vw;
    height: 100vh;
    display: flex;
    justify-content: center;
    padding: 150px 40px;
    position: relative;
    transition: opacity 1s;
`;

const Grid = styled.div`
    width: 950px;
    height: 350px;
    display: grid;
    grid-template-columns: repeat(auto-fill, 300px);
    column-gap: 25px;
    row-gap: 25px;
    justify-content: center;
    align-items: center;
    position: relative;
    margin-right: 40px;
    @media only screen and (max-width: 1407px) {
        width: 300px;
    }
    @media only screen and (min-width: 1408px) and (max-width: 1678px) {
        width: 625px;
    }
`;

const Filler = styled.div`
    width: 300px;
    height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    background-color: #000;
    color: #fff;
    font-size: 18px;
    text-align: right;
`;

interface IPrediction {
    segment: string;
    dataPractice: number;
}

interface IDataPractice {
    name: string;
    about: string;
    segments: string[];
}

type Props = {
    predictions: IPrediction[];
};

const initialDataPractices: IDataPractice[] = [
    {
        name: 'First Party Collection/Use',
        about: 'How and why a service provider collects user information.',
        segments: [],
    },
    {
        name: 'Third Party Sharing/Collection',
        about: 'How user information may be shared with or collected by third parties.',
        segments: [],
    },
    {
        name: 'User Choice/Control',
        about: 'Choices and control options available to users.',
        segments: [],
    },
    {
        name: 'User Access, Edit, & Deletion',
        about: 'If and how users may access, edit, or delete their information.',
        segments: [],
    },
    {
        name: 'Data Retention',
        about: 'How long user information is stored.',
        segments: [],
    },
    {
        name: 'Data Security',
        about: 'How user information is protected.',
        segments: [],
    },
    {
        name: 'Policy Change',
        about: 'If and how users will be informed about changes to the privacy policy.',
        segments: [],
    },
    {
        name: 'Do Not Track',
        about: 'If and how Do Not Track signals for online tracking and advertising are honoured.',
        segments: [],
    },
    {
        name: 'International & Specific Audiences',
        about: 'Practices that pertain only to specific group of users.',
        segments: [],
    },
];

const scrollToRef = (ref: React.RefObject<HTMLDivElement>) =>
    window.scrollTo({
        top: ref.current!.offsetTop,
        behavior: 'smooth',
    });

const Analysis = (props: Props) => {
    const [opacity, setOpacity] = useState(0);
    const [viewing, setViewing] = useState<IDataPractice | null>(null);
    const [closing, setClosing] = useState(false);
    const [dataPractices] = useState<IDataPractice[]>(initialDataPractices);
    const ref = useRef(null);

    useEffect(() => {
        // process predictions
        props.predictions.forEach((p) => {
            console.log(p.dataPractice);

            if (p.dataPractice >= 0 && p.dataPractice <= 5) {
                dataPractices[p.dataPractice].segments.push(p.segment);
            }
        });

        // scroll to analysis and make opaque
        if (opacity !== 1) {
            scrollToRef(ref);
            setOpacity(1);
        }
    }, []);

    // when the viewer is closing / closed grid width is handled by above media queries
    const gridStyle = viewing !== null || closing ? {} : { width: '100%', maxWidth: '1200px' };

    const numSegments = props.predictions.length;
    const numSegmentsColour = numSegments === 0 ? '#e53935' : '#1e88e5';

    return (
        <Container style={{ opacity }} ref={ref}>
            <Grid style={gridStyle}>
                <Filler>
                    <span>
                        Detected data practices for
                        <span style={{ color: numSegmentsColour }}>{` ${numSegments} `}</span>
                        segments.
                    </span>
                </Filler>

                {dataPractices.map((d) => (
                    <DataPractice name={d.name} about={d.about} onShow={() => setViewing(d)} />
                ))}
            </Grid>

            {viewing !== null && (
                <Viewer
                    dataPractice={viewing}
                    onHide={() => {
                        setClosing(true);

                        setTimeout(() => {
                            setClosing(false);
                            setViewing(null);
                        }, 200);
                    }}
                />
            )}
        </Container>
    );
};

export default Analysis;
