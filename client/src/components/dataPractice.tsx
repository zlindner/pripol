import React, { useState } from 'react';
import Popover, { ArrowContainer } from 'react-tiny-popover';
import styled from 'styled-components';

const Container = styled.div`
    width: 300px;
    height: 100px;
    position: relative;
    padding: 10px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);
`;

const Name = styled.span`
    width: 250px;
    display: inline-block;
`;

const About = styled.div`
    width: 20px;
    height: 20px;
    position: absolute;
    top: 10px;
    right: 10px;
    border-radius: 50%;
    text-align: center;
    cursor: pointer;

    &:hover {
        background-color: rgba(0, 0, 0, 0.15);
    }
`;

const AboutPopover = styled.div`
    width: 200px;
    padding: 10px;
    background-color: #ccc;
`;

type Props = {
    name: string;
    about: string;
};

const DataPractice = (props: Props) => {
    const [showAbout, setShowAbout] = useState(false);

    return (
        <Container>
            <Name>{props.name}</Name>

            <Popover
                isOpen={showAbout}
                position='top'
                align='start'
                onClickOutside={() => setShowAbout(false)}
                content={({ position, targetRect, popoverRect }) => (
                    <ArrowContainer position={position} targetRect={targetRect} popoverRect={popoverRect} arrowColor={'#ccc'} arrowSize={10}>
                        <AboutPopover>{props.about}</AboutPopover>
                    </ArrowContainer>
                )}
            >
                <About onMouseOver={() => setShowAbout(true)} onMouseLeave={() => setShowAbout(false)}>
                    ?
                </About>
            </Popover>
        </Container>
    );
};

export default DataPractice;
